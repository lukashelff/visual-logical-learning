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


def infer_symbolic(trainer, segmentation_similarity_threshold=.9, samples=1000):
    out_path = f'output/models/rcnn/inferred_symbolic/{trainer.settings}'
    all_labels = np.empty([0, 32], dtype=int)
    all_preds = np.empty([0, 32], dtype=int)
    model = trainer.model
    if trainer.dl is None:
        trainer.setup_ds(val_size=samples)
    dl = trainer.dl['val']
    # initialize tqdm progress bar
    prog_bar = tqdm(dl, total=len(dl))
    model.eval()
    model.to(trainer.device)
    for i, data in enumerate(prog_bar):
        images, targets = data
        images = list(image.to(trainer.device) for image in images)
        targets = [{k: v.to(trainer.device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            predictions = model(images)
        predictions = [{k: v.to(trainer.device) for k, v in t.items()} for t in predictions]
        for pred in predictions:
            train = preprocess_symbolics(pred, segmentation_similarity_threshold)
    #         all_labels = np.vstack((all_labels, labels))
    #         all_preds = np.vstack((all_preds, preds))
    # acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
    # print(f'fold {fold} acc score: {acc}')


def preprocess_symbolics(prediction, threshold=.9):
    labels = prediction["labels"]
    boxes = [i for i in prediction["boxes"]]
    masks = [i for i in prediction["masks"]]
    label_names = [rcnn_michalski_categories()[i] for i in prediction["labels"]]
    cars = sorted(labels[labels >= len(michalski_categories())])
    whole_car_masks = []
    car_numbers = []
    for car in cars:
        idx = labels.index(car)
        whole_car_masks.append(masks[idx])
        car_numbers.append(car - len(michalski_categories() + 1))
        del masks[idx]
        del labels[idx]

    train = np.zeros(len(cars) * 8)
    for mask, label_names in zip(masks, label_names):
        for car_number, whole_car_mask in zip(whole_car_masks, car_numbers):
            mask_pixel_sum = np.sum(mask)
            similarity = np.sum(mask * whole_car_mask) / mask_pixel_sum
            if similarity > threshold:
                class_int = michalski_categories().index(label_names)
                binary_class = np.zeros(22)
                binary_class[class_int] = 1
                label_int = class_to_label(class_int)
                idx = (car_number - 1) * 8 + label_int
                if idx == 5:
                    while train[idx] != 0 and idx < 6:
                        idx += 1
                if train[idx] != 0:
                    raise warnings.warn(
                        f"Overwriting {michalski_categories()[train[idx]]} with {michalski_categories()[class_int]} at index {idx} for car {car_number}.")
                train[idx] = class_int
    return train


def class_to_label(class_int):
    color = [0] * len(['yellow', 'green', 'grey', 'red', 'blue'])
    length = [1] * len(['short', 'long'])
    walls = [2] * len(["braced_wall", 'solid_wall'])
    roofs = [3] * len(["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof'])
    wheel_count = [4] * len(['2_wheels', '3_wheels'])
    load_obj = [5] * len(["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase'])
    all_labels = color + length + walls + roofs + wheel_count + load_obj
    return all_labels[class_int]


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
