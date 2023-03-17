import warnings

import numpy as np
import cv2
import torch
import glob as glob
import os
import time

from sklearn.metrics import accuracy_score
from tqdm import tqdm

from michalski_trains.dataset import michalski_categories, rcnn_michalski_categories
from models.rcnn.plot_prediction import plot_prediction


def infer_symbolic(trainer, segmentation_similarity_threshold=.9, samples=1000):
    out_path = f'output/models/rcnn/inferred_symbolic/{trainer.settings}'
    all_labels = []
    all_preds = []
    model = trainer.model
    if trainer.full_ds is None:
        trainer.setup_ds(val_size=samples)
    dl = trainer.dl['val']
    # initialize tqdm progress bar
    # prog_bar = tqdm(dl, total=len(dl))
    model.eval()
    model.to(trainer.device)
    ds = trainer.full_ds
    t_acc = []
    # for i, data in enumerate(prog_bar):
    for i in tqdm(range(samples)):
        image, target = ds.__getitem__(i)
        image = image.to(trainer.device).unsqueeze(0)
        labels = ds.get_attributes(i)
        with torch.no_grad():
            predictions = model(image)
        predictions = [{k: v.to(trainer.device) for k, v in t.items()} for t in predictions]
        acc = accuracy_score(target['labels'].to('cpu').numpy(), predictions[0]['labels'].to('cpu').numpy())
        t_acc.append(acc)
        for j, pred in enumerate(predictions):
            symbolic, issues = preprocess_symbolics(pred, segmentation_similarity_threshold)
            # if issues:
            #     plot_prediction(pred, i, image[j], device=trainer.device)
            all_preds.append(symbolic.to('cpu').numpy())
        all_labels.append(labels.to('cpu').numpy())

    # create numpy array with all predictions and labels
    b = np.zeros([len(all_preds), len(max(all_preds, key=lambda x: len(x)))])
    for i, j in enumerate(all_preds):
        b[i][0:len(j)] = j
    all_preds = b

    b = np.zeros([len(all_labels), len(max(all_labels, key=lambda x: len(x)))])
    for i, j in enumerate(all_labels):
        b[i][0:len(j)] = j
    all_labels = b

    acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    print(f'acc score: {acc}')
    print(f'average acc score: {np.mean(t_acc)}')


def preprocess_symbolics(prediction, threshold=.8):
    labels = prediction["labels"]
    boxes = prediction["boxes"]
    masks = prediction["masks"]
    scores = prediction["scores"]
    cars = sorted(labels[labels >= len(michalski_categories())].unique())
    # labels = labels.tolist()
    label_names = [rcnn_michalski_categories()[i] for i in labels]
    # get indices of all cars
    all_car_indices = []
    selected_car_indices = []
    issues = False
    for car in cars:
        indices = ((labels == car).nonzero(as_tuple=True)[0])
        indices = indices.to('cpu').tolist()
        all_car_indices += indices
        if len(indices) > 1:
            # select car with the highest score if there are multiple cars with same car number
            print(f"Multiple cars with same number: {len(indices)} cars with car number {rcnn_to_car_number(car)}."
                  f" Selecting car with highest prediction score.")
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
    train = torch.zeros(len(cars) * 8, dtype=torch.uint8)
    train_scores = torch.zeros(len(cars) * 8, dtype=torch.float32)
    for car_n, car_index in enumerate(selected_car_indices):
        whole_car_mask = masks[car_index]
        car_number = car_n + 1

        for attribute_indice in attribute_indices:
            mask = masks[attribute_indice]
            label = labels[attribute_indice]
            label_name = label_names[attribute_indice]
            similarity = get_similarity_score(mask, whole_car_mask)
            if similarity > threshold:
                # class_int = michalski_categories().index(label_name)
                # binary_class = np.zeros(22)
                # binary_class[label] = 1
                label_category = class_to_label(label)
                idx = (car_number - 1) * 8 + label_category
                if label_category == 5:
                    while train[idx] != 0 and (idx % 8) < 7:
                        idx += 1
                if train[idx] != 0:
                    if train[idx] == label:
                        print(f"Duplicate Mask: Mask for car {car_number} with label {label_name} was predicted twice.")
                        continue
                    elif train[idx] != label:
                        issues = True
                        if scores[attribute_indice] > train_scores[idx]:
                            print(f'Mask conflict: {michalski_categories()[train[idx]]} with score {train_scores[idx]} '
                                  f'and {michalski_categories()[label]} with score {scores[attribute_indice]}'
                                  f' for car {car_number}. Selecting higher score.')
                            train[idx] = label
                            train_scores[idx] = scores[attribute_indice]
                        else:
                            warnings.warn(
                                f'Mask conflict: {michalski_categories()[train[idx]]} with score {train_scores[idx]} '
                                f'and {michalski_categories()[label]} with score {scores[attribute_indice]}'
                                f' for car {car_number}. Selecting higher score.')
                else:
                    train[idx] = label
                    train_scores[idx] = scores[attribute_indice]
                # break
    return train, issues


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
    return label_val - len(michalski_categories()) + 1


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
