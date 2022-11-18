import json

import cv2
import matplotlib.colors as mcolors
import torchvision
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog, build_detection_test_loader, build_detection_train_loader,
)
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from torch.utils.data import random_split

from m_train_dataset import get_datasets
from models.detectron import register_ds
from util import *


def detectron_pred_vis_images(cfg, base_scene, train_col):
    model = build_model(cfg)
    path = cfg.OUTPUT_DIR

    model_path = f'./output/detectron/RandomTrains/{base_scene}/model_final.pth'
    if not os.path.isfile(model_path):
        raise AssertionError(f'trained detectron model for {train_col} in {base_scene} not found \n'
                             f'please consider to training a model first')
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(model_path)
    model.eval()
    image_count = 10
    register_ds(base_scene, train_col, image_count)
    metadata = MetadataCatalog.get("michalski_val_ds")
    data_loader = build_detection_test_loader(cfg, cfg.DATASETS.TEST[0])
    with torch.no_grad():
        for id, data in enumerate(data_loader):
            # im = torch2numpy(data[0]['image'])[:, :, ::-1]
            im_path = data[0]['file_name']
            im = cv2.imread(im_path)[:, :, ::-1]
            preds = model(data)
            instances = preds[0]['instances'].to("cpu")
            for c in range(len(instances)):
                visualizer = Visualizer(im, metadata=metadata, scale=2)
                instance = instances[c]
                label = metadata.thing_classes[int(instance.pred_classes)]
                draw = visualizer.draw_instance_predictions(instance)
                path = cfg.OUTPUT_DIR + f'/model_predictions/'
                os.makedirs(path, exist_ok=True)
                draw.save(path + f'train_{id}_instance_{c}_has_label_{label}.png')
                del visualizer
                # im = draw.get_image()
                # plt.imshow(im)
                # plt.title(f'train {id} predicted mask with label ' + label)
                # plt.savefig(path, bbox_inches='tight')


def plt_metrics(base_scene, train_col):
    path = f'output/detectron/{train_col}/{base_scene}'
    metrics_path = path + '/metrics.json'
    os.makedirs(path + '/statistics/', exist_ok=True)
    colors = list(mcolors.TABLEAU_COLORS.values())
    if not os.path.isfile(metrics_path):
        raise AssertionError(f'trained data for {train_col} in {base_scene} not found \n'
                             f'please consider to training a model first')
    with open(metrics_path, 'r') as f:
        metrics = [json.loads(line) for line in f]
        cls_accuracy = [metric["fast_rcnn/cls_accuracy"] for metric in metrics]
        fg_cls_accuracy = [metric["fast_rcnn/fg_cls_accuracy"] for metric in metrics]
        iteration = [metric["iteration"] for metric in metrics]
        loss_box_reg = [metric["loss_box_reg"] for metric in metrics]
        loss_cls = [metric["loss_cls"] for metric in metrics]
        loss_mask = [metric["loss_mask"] for metric in metrics]
        loss_rpn_cls = [metric["loss_rpn_cls"] for metric in metrics]
        loss_rpn_loc = [metric["loss_rpn_loc"] for metric in metrics]
        lr = [metric["lr"] for metric in metrics]
        accuracy = [metric["mask_rcnn/accuracy"] for metric in metrics]
        num_bg_samples = [metric["roi_head/num_bg_samples"] for metric in metrics]
        num_fg_samples = [metric["roi_head/num_fg_samples"] for metric in metrics]
        num_neg_anchors = [metric["rpn/num_neg_anchors"] for metric in metrics]
        num_pos_anchors = [metric["rpn/num_pos_anchors"] for metric in metrics]
        total_loss = [metric["total_loss"] for metric in metrics]

        # plot accuracy#
        # plt.title(f'detectron acc on {train_col} in {base_scene}')
        plt.plot(iteration, cls_accuracy, label=f'cls_accuracy {round(cls_accuracy[-1] * 100, 1)}%')
        plt.plot(iteration, fg_cls_accuracy, label=f'fg_cls_accuracy {round(fg_cls_accuracy[-1] * 100, 1)}%')
        plt.plot(iteration, accuracy, label=f'accuracy {round(fg_cls_accuracy[-1] * 100, 1)}%')
        plt.xlabel("Training iteration")
        plt.ylabel("Accuracy")
        plt.legend(loc="lower right")
        plt.savefig(path + f'/statistics/accuracy.png', dpi=400)
        plt.close()

        # plot accuracy#
        plt.title(f'detectron loss on {train_col} in {base_scene}')
        losses = [loss_box_reg, loss_cls, loss_mask, loss_rpn_cls, loss_rpn_loc, total_loss]
        plt.plot(iteration, loss_box_reg, label=f'loss_box_reg {round(loss_box_reg[-1], 2)}')
        plt.plot(iteration, loss_cls, label=f'loss_cls {round(loss_cls[-1], 2)}')
        plt.plot(iteration, loss_mask, label=f'loss_mask {round(loss_mask[-1], 2)}')
        plt.plot(iteration, loss_rpn_cls, label=f'loss_rpn_cls {round(loss_rpn_cls[-1], 2)}')
        plt.plot(iteration, loss_rpn_loc, label=f'loss_rpn_loc {round(loss_rpn_loc[-1], 2)}')
        plt.plot(iteration, total_loss, label=f'total_loss {round(total_loss[-1], 2)}')
        plt.xlabel("iteration")
        plt.ylabel("loss")
        plt.legend(loc="upper right")
        plt.savefig(path + f'/statistics/loss.png', dpi=400)
        plt.close()
