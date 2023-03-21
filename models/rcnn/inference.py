import warnings

import numpy as np
import cv2
import torch
import glob as glob
import os
import time

from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange

from michalski_trains.dataset import michalski_categories, rcnn_michalski_categories
from models.rcnn.plot_prediction import plot_prediction


def infer_symbolic(trainer, segmentation_similarity_threshold=.8, samples=1000):
    out_path = f'output/models/rcnn/inferred_symbolic/{trainer.settings}'
    all_labels = []
    all_preds = []
    model = trainer.model
    if trainer.full_ds is None:
        trainer.setup_ds(val_size=samples)
    dl = trainer.dl['val']
    model.eval()
    model.to(trainer.device)
    ds = trainer.full_ds
    t_acc = []

    for i in range(samples):
        image, target = ds.__getitem__(i)
        image = image.to(trainer.device).unsqueeze(0)
        labels = ds.get_attributes(i).to('cpu').numpy()
        with torch.no_grad():
            output = model(image)
        output = [{k: v.to(trainer.device) for k, v in t.items()} for t in output]
        symbolic, issues = process_symbolics(output[0], segmentation_similarity_threshold)
        symbolic = symbolic.to('cpu').numpy()
        length = max(len(symbolic), len(labels))
        symbolic = np.pad(symbolic, (0, length - len(symbolic)), 'constant', constant_values=0)
        labels = np.pad(labels, (0, length - len(labels)), 'constant', constant_values=0)
        if '[' in issues:
            plot_prediction(output[0], i, image[0], device=trainer.device)
        all_labels.append(labels)
        all_preds.append(symbolic)
        train_acc = accuracy_score(labels, symbolic)
        t_acc.append(train_acc)
        out_text = f"image {i}/{samples}, accuracy score: {round(train_acc * 100, 1)}%, " \
                   f"running accuracy score: {(np.mean(t_acc) * 100).round(3)}%, Number of gt attributes {len(labels[labels > 0])}. "
        print(out_text + issues)
        out_text = out_text

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


def process_symbolics(prediction, threshold=.8):
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
    issues = ""
    for car in cars:
        indices = ((labels == car).nonzero(as_tuple=True)[0])
        indices = indices.to('cpu').tolist()
        all_car_indices += indices
        if len(indices) > 1:
            # select car with the highest score if there are multiple cars with same car number
            issues += f"Multiple cars with same number: {len(indices)} cars with car number {rcnn_to_car_number(car)}." \
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
    train = torch.zeros(len(cars) * 8, dtype=torch.uint8)
    train_scores = torch.zeros(len(cars) * 8, dtype=torch.float32)
    skipped_indicies = []
    for attribute_index in attribute_indices:
        allocated = False
        label = labels[attribute_index]
        label_name = label_names[attribute_index]
        car_similarities = get_all_similarities_score(masks, car_indices=selected_car_indices,
                                                      attribute_index=attribute_index)
        car_number = np.argmax(car_similarities) + 1
        similarity = car_similarities[car_number - 1]

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
                    issues += f"Duplicate Mask: Mask for car {car_number} with label {label_name} was predicted twice."
                    continue
                elif train[idx] != label:
                    issues = True
                    if scores[attribute_index] > train_scores[idx]:
                        issues += f'Mask conflict: {michalski_categories()[train[idx]]} with score' \
                                  f' {train_scores[idx]} and {michalski_categories()[label]} with score' \
                                  f' {scores[attribute_index].round(3)} for car {car_number}. Selecting label with higher score.'
                        train[idx] = label
                        allocated = True
                        train_scores[idx] = scores[attribute_index]
                    else:
                        issues += f'Mask conflict: {michalski_categories()[train[idx]]} with score ' \
                                  f'{train_scores[idx]} and {michalski_categories()[label]} with score' \
                                  f' {scores[attribute_index].round(3)} for car {car_number}. Selecting label with higher score.'
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
    issues = f'Number of predictions: {nr_output_attr}, number of found allocations: {nr_attr_processed}, ' \
             f'not allocated label number: {skipped_indicies}. ' \
             + issues
    return train, issues


def get_all_similarities_score(masks, attribute_index, car_indices):
    car_similarities = []
    for car_n, car_index in enumerate(car_indices):
        whole_car_mask = masks[car_index]
        mask = masks[attribute_index]
        similarity = get_similarity_score(mask, whole_car_mask)
        car_similarities += [similarity]
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
