import copy
import json
import logging
import os
import pprint
from collections import OrderedDict

import detectron2.data.transforms as T
import detectron2.utils.comm as comm
import torch
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data import detection_utils as utils
from detectron2.engine import default_writers
from detectron2.evaluation import (
    DatasetEvaluators,
    inference_on_dataset,
    print_csv_format, DatasetEvaluator,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage
from rtpt import RTPT
from torch.utils.data import random_split

from blender_image_generator.json_util import encodeMask
import m_train_dataset
from pycocotools import mask as maskUtils

logger = logging.getLogger("detectron2")


def setup(path, base_scene, train_col):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(path)
    cfg.OUTPUT_DIR = f'./output/detectron/{train_col}/{base_scene}'
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    cfg.freeze()
    return cfg

def register_ds(base_scene, train_col, image_count):
    y_val = 'mask'
    full_ds = m_train_dataset.get_datasets(base_scene, train_col, image_count, y_val)
    train_size, val_size = int(0.7 * image_count), int(0.3 * image_count)
    train_dataset, val_dataset = random_split(full_ds, [train_size, val_size])

    def create_train_ds():
        return create_michalski_train_ds(train_dataset.indices)

    def create_val_ds():
        return create_michalski_train_ds(val_dataset.indices)

    def create_full_ds():
        return create_michalski_train_ds([*range(image_count)])

    def create_michalski_train_ds(ds_ind):
        ds = []
        height, width = 270, 480
        all_categories = full_ds.attribute_classes[1:]
        for id, index in enumerate(ds_ind):
            train = full_ds.get_m_train(index)
            train_mask = full_ds.get_mask(index)
            image_pth = full_ds.get_image_path(index)
            data_dict = {
                'file_name': image_pth,
                'height': height,
                'width': width,
                'image_id': id,
                'annotations': []
            }
            for (car_name, car_mask), car in zip(train_mask.items(), train.get_cars()):
                position = car_name
                whole_car_mask = car_mask['mask']
                # title = f'{car_name} segmentation'
                # plot_rle(whole_car_mask, title)

                whole_car_bbox = maskUtils.toBbox(whole_car_mask)
                del car_mask['mask'], car_mask['b_box'], car_mask['world_cord']
                for att_name, att in car_mask.items():
                    label = att['label']
                    if label != 'none':
                        annotations = {}

                        if att_name == 'length' or att_name == 'color':
                            rle, bbox = whole_car_mask, whole_car_bbox
                        else:
                            rle = att['mask']
                            bbox = maskUtils.toBbox(rle)
                        # bbox = boxes.Boxes(torch.tensor(bbox).unsqueeze(dim=0))
                        annotations['bbox'] = list(bbox)
                        annotations['bbox_mode'] = 1
                        annotations['category_id'] = all_categories.index(label)
                        annotations['segmentation'] = rle
                        data_dict['annotations'].append(annotations)
            ds.append(data_dict)
        return ds

    def mapper(dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # can use other ways to read image
        image = utils.read_image(dataset_dict["file_name"], format="RGB")
        # image = Image.open(dataset_dict["file_name"]).convert('RGB')
        # See "Data Augmentation" tutorial for details usage
        auginput = T.AugInput(image)
        transform = auginput
        image = torch.from_numpy(auginput.image.transpose(2, 0, 1))
        annos = [
            utils.transform_instance_annotations(annotation, [transform], image.shape[1:])
            for annotation in dataset_dict.pop("annotations")
        ]
        return {
            # create the format that the model expects
            "image": image,
            "instances": utils.annotations_to_instances(annos, image.shape[1:])
        }

    DatasetCatalog.register("michalski_train_ds", create_train_ds)
    DatasetCatalog.register("michalski_val_ds", create_val_ds)
    DatasetCatalog.register("michalski_ds", create_full_ds)
    all_att = full_ds.attribute_classes[1:]
    MetadataCatalog.get("michalski_train_ds").thing_classes = all_att
    MetadataCatalog.get("michalski_val_ds").thing_classes = all_att
    MetadataCatalog.get("michalski_ds").thing_classes = all_att
    # data = DatasetCatalog.get("michalski_val_ds")


def do_train(cfg, model, experiment_name, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
            checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )
    rtpt = RTPT(name_initials='LH', experiment_name=experiment_name,
                max_iterations=cfg.SOLVER.MAX_ITER - start_iter + 1)
    rtpt.start()

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    print(f'start at iteration {start_iter}, complete at iteration {max_iter}')
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            rtpt.step()
            storage.iter = iteration

            loss_dict = model(data)
            losses = sum(loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                    cfg.TEST.EVAL_PERIOD > 0
                    and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                    and iteration != max_iter - 1
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and ((iteration + 1) % 20 == 0 or iteration == max_iter - 1):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def do_test(cfg, base_scene, train_col):
    model = build_model(cfg)
    out_dir = cfg.OUTPUT_DIR
    # model_path = out_dir + '/model_final.pth'
    model_path = f'./output/detectron/RandomTrains/{base_scene}/model_final.pth'
    if not os.path.isfile(model_path):
        raise ValueError(
            f'trained detectron model for {train_col} in {base_scene} not found \n please consider to training a model first')
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)
    model.eval()
    image_count = 10000
    register_ds(base_scene, train_col, image_count)
    dataset_name = cfg.DATASETS.TEST[0]

    results = OrderedDict()
    data_loader = build_detection_test_loader(cfg, dataset_name)
    evaluator = get_evaluator(
        cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
    )
    results_i = inference_on_dataset(model, data_loader, DatasetEvaluator())
    print(results_i)
    json_pth = f'./output/detectron/RandomTrains/{base_scene}/model_test.json'
    with open(json_pth, 'w+') as f:
        json.dump(results_i, f, indent=4)
    results[dataset_name] = results_i
    if comm.is_main_process():
        logger.info("Evaluation results for {} in csv format:".format(dataset_name))
        print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    # json_pth = f'./output/detectron/RandomTrains/{base_scene}/model_test.json'
    # with open(json_pth, 'w+') as f:
    #     json.dump(results, f, indent=4)
    return results


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_list.append(DatasetEvaluator())
    # evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    # if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
    #     evaluator_list.append(
    #         SemSegEvaluator(
    #             dataset_name,
    #             distributed=True,
    #             output_dir=output_folder,
    #         )
    #     )
    # if evaluator_type in ["coco", "coco_panoptic_seg"]:
    #     evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    # if evaluator_type == "coco_panoptic_seg":
    #     evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    # if evaluator_type == "cityscapes_instance":
    #     return CityscapesInstanceEvaluator(dataset_name)
    # if evaluator_type == "cityscapes_sem_seg":
    #     return CityscapesSemSegEvaluator(dataset_name)
    # if evaluator_type == "pascal_voc":
    #     return PascalVOCDetectionEvaluator(dataset_name)
    # if evaluator_type == "lvis":
    #     return LVISEvaluator(dataset_name, cfg, True, output_folder)
    # if len(evaluator_list) == 0:
    #     raise NotImplementedError(
    #         "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
    #     )
    # if len(evaluator_list) == 1:
    #     return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def predict_instances(base_scene, train_col, cfg):
    model = build_model(cfg)
    out_dir = cfg.OUTPUT_DIR
    # model_path = out_dir + '/model_final.pth'
    model_path = f'./output/detectron/RandomTrains/{base_scene}/model_final.pth'
    if not os.path.isfile(model_path):
        raise ValueError(
            f'trained detectron model for {train_col} in {base_scene} not found \n please consider to training a model first')
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)
    model.eval()
    image_count = 10000
    register_ds(base_scene, train_col, image_count)
    metadata = MetadataCatalog.get("michalsk_ds")
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[1])
    rtpt = RTPT(name_initials='LH', experiment_name=f'Pred_In_{base_scene[:3]}_{train_col[0]}',
                max_iterations=len(data_loader))
    rtpt.start()
    with torch.no_grad():
        object_masks = {}
        for data in data_loader:
            rtpt.step()
            preds = model(data)
            # preds[0]['instances'][0].pred_boxes
            # preds[0]['instances'][0].scores
            # preds[0]['instances'][0].pred_classes
            # preds[0]['instances'][0].pred_masks
            for pred, image in zip(preds, data):
                file_name = image['file_name']
                image_name = file_name.split("/")[-1]
                image_id = image['image_id']
                instances = pred['instances'].to("cpu")
                obj_mask = {
                    'instances': {},
                    'file_name': file_name
                }
                for i in range(len(instances)):
                    instance = instances[i]
                    pred_box = instance.pred_boxes
                    score = float(instance.scores)
                    pred_class = int(instance.pred_classes)
                    pred_mask = instance.pred_masks.squeeze().numpy()
                    obj_mask['instances'][i] = {}
                    obj_mask['instances'][i]['pred_box'] = tuple(pred_box.tensor.squeeze().int().tolist())
                    obj_mask['instances'][i]['score'] = score
                    ##############################################
                    # obj_mask['instances'][i]['pred_class'] = pred_class + 1???
                    ##############################################
                    obj_mask['instances'][i]['pred_class'] = pred_class
                    # obj_mask['instances'][i]['pred_class_name'] = metadata.thing_classes[pred_class]
                    obj_mask['instances'][i]['pred_mask'] = encodeMask(pred_mask)
                object_masks[image_id] = obj_mask
    path = f'./output/detectron/{train_col}/{base_scene}'
    os.makedirs(path, exist_ok=True)
    with open(path + '/predictions.json', 'w+') as fp:
        json.dump(object_masks, fp, indent=2)