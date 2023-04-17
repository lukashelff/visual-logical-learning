import time

import cv2
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import (
    MetadataCatalog, build_detection_test_loader, )
from detectron2.modeling import build_model
from detectron2.utils.visualizer import Visualizer
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from michalski_trains.dataset import michalski_labels, blender_categories, original_categories, rcnn_blender_categories
from michalski_trains.m_train import BlenderCar, MichalskiTrain
from models.rcnn.plot_prediction import plot_mask
from util import *


def infer_symbolic(model, dl, device, segmentation_similarity_threshold=.8, samples=None, debug=False):
    '''
    Infer symbolic representation of the scene
    :param model: model to infer
    :param dl: dataloader
    :param device: device to run on
    :param segmentation_similarity_threshold: threshold for similarity between two segments
    :param samples: number of samples to infer
    :param debug: debug mode
    :return: all_labels - 2d ndarray (samples, attributes) of ground truth labels
             all_preds - 2d ndarray (samples, attributes) of predicted labels
             avg_acc - average accuracy over all samples and all attributes (0-1)


    '''
    # out_path = f'output/models/rcnn/inferred_symbolic/{trainer.settings}'
    all_labels = []
    all_preds = []
    # if trainer.full_ds is None:
    #     trainer.setup_ds(val_size=samples)
    samples = len(dl.dataset) if samples is None else samples
    model.eval()
    model.to(device)
    train_accuracies = []
    indices = dl.dataset.indices[:samples]
    ds = dl.dataset.dataset
    prog_bar = tqdm(indices, total=len(indices))
    m_labels = michalski_labels()

    for i in tqdm(prog_bar):

        image, target = ds.__getitem__(i)
        image = image.to(device).unsqueeze(0)
        labels = ds.get_attributes(i).to('cpu').numpy()
        if debug:
            plot_mask(target, i, image[0], tag='gt')

        with torch.no_grad():
            output = model(image)
        output = [{k: v.to(device) for k, v in t.items()} for t in output]
        symbolic, issues = prediction_to_symbolic(output[0], segmentation_similarity_threshold)
        symbolic = symbolic.to('cpu').numpy()
        length = max(len(symbolic), len(labels))
        symbolic = np.pad(symbolic, (0, length - len(symbolic)), 'constant', constant_values=0)
        labels = np.pad(labels, (0, length - len(labels)), 'constant', constant_values=0)
        # if debug:
        #     plot_mask(output[0], i, image[0], tag='prediction')
        all_labels.append(labels)
        all_preds.append(symbolic)
        accuracy = accuracy_score(labels, symbolic)
        train_accuracies.append(accuracy)
        debug_text = f"image {i}/{samples}, accuracy score: {round(accuracy * 100, 1)}%, " \
                     f"running accuracy score: {(np.mean(train_accuracies) * 100).round(3)}%, " \
                     f"Number of gt attributes {len(labels[labels > 0])}. "

        prog_bar.set_description(desc=f"Acc: {(np.mean(train_accuracies) * 100).round(3)}")

        if debug:
            print(debug_text + issues)

    # create numpy array with all predictions and labels
    # pad with zeros to make all arrays the same length
    max_train_labels = max(len(max(all_preds, key=lambda x: len(x))), len(max(all_labels, key=lambda x: len(x))))
    preds_padded, labels_padded = np.zeros([len(all_preds), max_train_labels]), np.zeros(
        [len(all_labels), max_train_labels])
    if len(all_preds) != len(all_labels):
        raise ValueError(f'Number of predictions and labels does not match: {len(all_preds)} != {len(all_labels)}')
    for i, (p, l) in enumerate(zip(all_preds, all_labels)):
        preds_padded[i][0:len(p)] = p
        labels_padded[i][0:len(l)] = l

    average_acc = accuracy_score(preds_padded.flatten(), labels_padded.flatten())
    txt = f'average accuracy over all symbols: {round(average_acc, 3)}, '
    label_acc = 'label accuracies:'
    # labels_car, car_predictions = labels_padded.reshape((-1, 8)), preds_padded.reshape((-1, 8))
    for label_id, label in enumerate(m_labels):
        lab = labels_padded.reshape((-1, 8))[:, label_id]
        pred = preds_padded.reshape((-1, 8))[:, label_id]
        acc = accuracy_score(lab[lab > 0], pred[lab > 0])
        label_acc += f' {label}: {round(acc * 100, 3)}%'
    print(txt + label_acc)

    # print(f'average train acc score: {np.mean(t_acc).round(3)}')
    return preds_padded, labels_padded, average_acc, np.mean(train_accuracies)


def infer_dataset(model, dl, device, out_dir):
    rcnn_symbolics, _, _, _ = infer_symbolic(model, dl, device, debug=False)
    ds_labels = ['west', 'east']
    train_labels = [ds_labels[dl.dataset.dataset.get_direction(i)] for i in range(dl.dataset.dataset.__len__())]
    trains = rcnn_decode(train_labels, rcnn_symbolics)
    from ilp.dataset_functions import create_bk
    create_bk(trains, out_dir)
    print('rcnn inferred symbolic saved to: ', out_dir)


def rcnn_decode(train_labels, rcnn_symbolics):
    trains = []
    prog_bar = tqdm(range(len(rcnn_symbolics)), total=len(rcnn_symbolics), desc='converting symbolic')
    for s_i in prog_bar:
        symbolic = rcnn_symbolics[s_i].reshape(-1, 8)
        train = int_encoding_to_michalski_symbolic(symbolic)
        cars = []
        for car_id, car in enumerate(train):
            car = BlenderCar(*car)
            cars.append(car)
        train = MichalskiTrain(cars, train_labels[s_i], 0)
        trains.append(train)
    return trains


def prediction_to_symbolic(prediction, threshold=.8):
    '''
    Convert prediction to symbolic representation
    :param prediction: prediction from model
    :param threshold: threshold for similarity between two segments
    :return: symbolic representation of the scene
    '''
    labels = prediction["labels"]
    boxes = prediction["boxes"]
    masks = prediction["masks"]
    scores = prediction["scores"].to('cpu').tolist()
    cars = sorted(labels[labels >= len(blender_categories())].unique())
    # labels = labels.tolist()
    label_names = []
    for l in labels:
        cats = rcnn_blender_categories()
        if l < len(cats):
            label_names.append(cats[l])
        else:
            label_names.append(f'car_{rcnn_to_car_number(l)}')

    # get indices of all cars
    all_car_indices = []
    # select valid cars
    selected_car_indices = []
    debug_info = ""
    # get all cars, select car with highest score if there are multiple cars with same car number, others are discarded
    for car in cars:
        indices = ((labels == car).nonzero(as_tuple=True)[0])
        indices = indices.to('cpu').tolist()
        all_car_indices += indices
        if len(indices) > 1:
            # select car with the highest score if there are multiple cars with same car number
            debug_info += f"Multiple cars with same number: {len(indices)} cars with car number {rcnn_to_car_number(car)}." \
                          f" Selecting car with highest prediction score."
            idx = indices[0]
            # issues = True
            for i in indices[1:]:
                if scores[i] > scores[idx]:
                    idx = i
        else:
            idx = indices[0]
        selected_car_indices.append(idx)
    # get indices of all attributes
    attribute_indices = [i for i in range(len(labels)) if i not in all_car_indices]


    shape = ['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped']
    length = ['short', 'long']
    walls = ["double", 'not_double']
    roofs = ['arc', 'flat', 'jagged', 'peaked']
    wheel_count = ['2', '3']
    load_obj = ["rectangle", "triangle", 'circle', 'diamond', 'hexagon', 'utriangle']
    original_categories = ['none'] + shape + length + walls + roofs + wheel_count + load_obj
    # initialize symbolic representation
    car_init = [0, 6, 8, 0, 14, 0, 0, 0]
    train = torch.tensor(car_init * len(cars), dtype=torch.uint8)
    # train = torch.zeros(len(cars) * 8, dtype=torch.uint8)
    train_scores = [0] * len(cars) * 8
    skipped_indicies = []
    # iterate over all predicted attributes
    for attribute_index in attribute_indices:
        allocated = False
        label = labels[attribute_index]
        label_name = label_names[attribute_index]
        car_similarities = get_all_similarities_score(masks, car_indices=selected_car_indices,
                                                      attribute_index=attribute_index)
        if len(car_similarities) == 0:
            debug_info += f"Attribute {label_name} not allocated to any car. "
            continue
        car_number = np.argmax(car_similarities) + 1
        similarity = car_similarities[car_number - 1]
        # if similarity higher than threshold allocate attribute to car
        if similarity > threshold:
            # class_int = blender_categories().index(label_name)
            # binary_class = np.zeros(22)
            # binary_class[label] = 1
            label_category = class_to_label(label)
            idx = (car_number - 1) * 8 + label_category
            # if attribute is a payload, check if there is already a payload allocated to the car
            if label_category == 5:
                # todo: sort payload by score to replace payload with lower score if there are to many payloads
                while train[idx] != 0 and (idx % 8) < 7:
                    idx += 1
            if train[idx] != 0:
                if train[idx] == label:
                    debug_info += f"Duplicate Mask: Mask for car {car_number} with label {label_name} was predicted twice."
                elif train[idx] != label:
                    if scores[attribute_index] > train_scores[idx]:
                        debug_info += f'Mask conflict: {blender_categories()[train[idx]]} with score' \
                                      f' {round(train_scores[idx], 3)} and {blender_categories()[label]} with score' \
                                      f' {round(scores[attribute_index], 3)} for car {car_number}. Selecting label with higher score.'
                        train[idx] = label
                        allocated = True
                        train_scores[idx] = scores[attribute_index]
                    else:
                        debug_info += f'Mask conflict: {blender_categories()[train[idx]]} with score ' \
                                      f'{round(train_scores[idx], 3)} and {blender_categories()[label]} with score' \
                                      f' {round(scores[attribute_index], 3)} for car {car_number}. Selecting label with higher score.'
            else:
                train[idx] = label
                allocated = True
                train_scores[idx] = scores[attribute_index]
            # break
        if not allocated:
            skipped_indicies.append(attribute_index)
    nr_attr_processed = len(train[(train > 0) & (train < 22)])
    model_label_prdictions = prediction['labels'].to('cpu').numpy()
    nr_output_attr = len(model_label_prdictions[(model_label_prdictions > 0) & (model_label_prdictions < 22)])
    debug_info = f'Number of predictions: {nr_output_attr}, number of found allocations: {nr_attr_processed}, ' \
                 f'not allocated label number: {skipped_indicies}. ' \
                 + debug_info
    return train, debug_info


def get_all_similarities_score(masks, attribute_index, car_indices):
    car_similarities = []
    for car_n, car_index in enumerate(car_indices):
        whole_car_mask = masks[car_index]
        mask = masks[attribute_index]
        similarity = get_similarity_score(mask, whole_car_mask)
        car_similarities += [similarity.item()]
    return car_similarities


def get_similarity_score(mask, whole_car_mask):
    # determine to which degree mask is included in whole car mask
    # calculate similarity value by summing up all values in mask where mask is smaller than whole car mask
    # and summing up all values of whole car mask where mask is higher than whole car mask
    similarity = mask[mask <= whole_car_mask].sum() + whole_car_mask[mask > whole_car_mask].sum()
    similarity = similarity / mask.sum()

    # calculate difference between mask and whole car mask for values where mask is higher than whole car mask
    # asimilarity = mask[mask > whole_car_mask].sum() - whole_car_mask[mask > whole_car_mask].sum()
    # similarity = 1 - asimilarity / mask.sum()

    # calculate similarity by multiplication of mask and whole car mask, problem because we hve float values
    # when mask = whole car mask = 0.3 => similarity = 0.3 * 0.3 = 0.09 => similarity is too low
    # simi = mask * whole_car_mask
    # similarity = simi.sum() / mask.sum()
    return similarity


def rcnn_to_car_number(label_val):
    return label_val - len(blender_categories()) + 1


def class_to_label(class_int):
    none = [-1] * len(['none'])
    color = [0] * len(['yellow', 'green', 'grey', 'red', 'blue'])
    length = [1] * len(['short', 'long'])
    walls = [2] * len(["braced_wall", 'solid_wall'])
    roofs = [3] * len(["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof'])
    wheel_count = [4] * len(['2_wheels', '3_wheels'])
    load_obj = [5] * len(["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase'])
    all_labels = none + color + length + walls + roofs + wheel_count + load_obj
    return all_labels[class_int]


def label_type(idx):
    l = idx % 8
    labels = ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2', 'load_obj3']
    return labels[l]


def inference(model, images, device, classes, detection_threshold=0.8):
    model.to(device).eval()
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

    # to count the total number of images iterated through
    frame_count = 0
    # to keep adding the FPS for each image
    total_fps = 0
    for i in range(len(images)):
        # get the image file name for saving output later on
        # orig_image = image.copy()
        # # BGR to RGB
        # image = cv2.cvtColor(orig_image, cv2.COLOR_BGR2RGB).astype(np.float32)
        # # make the pixel range between 0 and 1
        # image /= 255.0
        # # bring color channels to front
        # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        # # convert to tensor
        # image = torch.tensor(image, dtype=torch.float).cuda()
        # # add batch dimension
        # image = torch.unsqueeze(image, 0)

        # torch image to cv2 image
        ori_image = images[i].permute(1, 2, 0).numpy()
        orig_image = cv2.cvtColor(ori_image, cv2.COLOR_RGB2BGR)

        start_time = time.time()
        with torch.no_grad():
            outputs = model(images[i].to(device))
        end_time = time.time()

        # get the current fps
        # fps = 1 / (end_time - start_time)
        # # add `fps` to `total_fps`
        # total_fps += fps
        # # increment frame count
        # frame_count += 1
        # # load all detection to CPU for further operations
        outputs = [{k: v.to('cpu') for k, v in t.items()} for t in outputs]
        # carry further only if there are detected boxes
        if len(outputs[0]['boxes']) != 0:
            boxes = outputs[0]['boxes'].data.numpy()
            scores = outputs[0]['scores'].data.numpy()
            # filter out boxes according to `detection_threshold`
            boxes = boxes[scores >= detection_threshold].astype(np.int32)
            draw_boxes = boxes.copy()
            # get all the predicited class names
            pred_classes = [classes[i] for i in outputs[0]['labels'].cpu().numpy()]

            # draw the bounding boxes and write the class name on top of it
            for j, box in enumerate(draw_boxes):
                class_name = pred_classes[j]
                color = COLORS[classes.index(class_name)]
                cv2.rectangle(orig_image,
                              (int(box[0]), int(box[1])),
                              (int(box[2]), int(box[3])),
                              color, 2)
                cv2.putText(orig_image, class_name,
                            (int(box[0]), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color,
                            2, lineType=cv2.LINE_AA)

            cv2.imshow('Prediction', orig_image)
            cv2.waitKey(1)
            cv2.imwrite(f"output/models/rcnn/inference_outputs/images/image_{i}.jpg", orig_image)
        print(f"Image {i + 1} done...")
        print('-' * 50)

    print('TEST PREDICTIONS COMPLETE')
    cv2.destroyAllWindows()
    # calculate and print the average FPS
    avg_fps = total_fps / frame_count
    print(f"Average FPS: {avg_fps:.3f}")


def int_encoding_to_michalski_symbolic(int_encoding: np.ndarray) -> [[str]]:
    '''
    Convert int encoding to original Michalski trains representation
    :param int_encoding: int encoding numpy array of size (n, 8), where n is the number of cars
                    1st column: color
                    2nd column: length
                    3rd column: wall
                    4th column: roof
                    5th column: wheels
                    6th column: load1
                    7th column: load2
                    8th column: load3
    :return: original michalski representation List[List[str]] of size (n, 8), where n is the number of cars
                    1st column: car_id
                    2nd column: shape
                    3rd column: length
                    4th column: double
                    5th column: roof
                    6th column: wheels
                    7th column: l_shape
                    8th column: l_num
    '''

    shape = ['rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped']
    length = ['short', 'long']
    walls = ["double", 'not_double']
    roofs = ['arc', 'flat', 'jagged', 'peaked']
    wheel_count = ['2', '3']
    load_obj = ["rectangle", "triangle", 'circle', 'diamond', 'hexagon', 'utriangle']
    original_categories = ['none'] + shape + length + walls + roofs + wheel_count + load_obj

    label_dict = {
        'shape': ['none', 'rectangle', 'bucket', 'ellipse', 'hexagon', 'u_shaped'],
        'length': ['short', 'long'],
        'walls': ["double", 'not_double'],
        'roofs': ['none', 'arc', 'flat', 'jagged', 'peaked'],
        'wheel_count': ['2', '3'],
        'load_obj': ["rectangle", "triangle", 'circle', 'diamond', 'hexagon', 'utriangle'],
    }

    int_encoding = int_encoding.astype(int)
    michalski_train = []
    for car_id, car in enumerate(int_encoding):
        if sum(car) > 0:
            n = str(car_id)
            shape = original_categories[car[0]]
            length = original_categories[car[1]]
            double = original_categories[car[2]]
            roof = original_categories[car[3]]
            wheels = original_categories[car[4]]
            l_shape = original_categories[car[5]]
            l_num = sum(car[5:] != 0)
            michalski_train += [[n, shape, length, double, roof, wheels, l_shape, l_num]]
    return michalski_train
