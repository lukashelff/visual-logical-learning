import time

import cv2
from rtpt import RTPT
from sklearn.metrics import accuracy_score
from tqdm import tqdm

from michalski_trains.dataset import michalski_labels, blender_categories
from michalski_trains.m_train import BlenderCar, MichalskiTrain
from models.rcnn.plot_prediction import plot_mask
from models.rcnn.template_matcher import prediction_to_symbolic_v2
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
    labs = ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2', 'load_obj3']
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
    prog_bar = tqdm(indices, total=len(indices), disable=debug)
    m_labels = michalski_labels()
    rtpt = RTPT(name_initials='LH', experiment_name='RCNN_infer_symbolic', max_iterations=len(indices))
    rtpt.start()
    for i in tqdm(prog_bar, disable=debug):
        image = ds.get_image(i)
        image = image.to(device).unsqueeze(0)
        labels = ds.get_attributes(i).to('cpu').numpy()
        # if debug:
        #     image, target = ds.__getitem__(i)
        #     plot_mask(target, i, image[0], tag='gt')
        with torch.no_grad():
            output = model(image)
        output = [{k: v.to(device) for k, v in t.items()} for t in output]
        # symbolic, issues = prediction_to_symbolic(output[0], segmentation_similarity_threshold)
        symbolic, issues = prediction_to_symbolic_v2(output[0], ds.get_ds_classes(), segmentation_similarity_threshold)
        symbolic = symbolic.to('cpu').numpy()
        length = max(len(symbolic), len(labels))
        symbolic = np.pad(symbolic, (0, length - len(symbolic)), 'constant', constant_values=0)
        labels = np.pad(labels, (0, length - len(labels)), 'constant', constant_values=0)

        all_labels.append(labels)
        all_preds.append(symbolic)
        accuracy = accuracy_score(labels, symbolic)
        if accuracy != 1:
            debug_text = f'image {i} incorrect labels, '

            for t_number, train in enumerate((labels == symbolic).reshape((-1, 8))):
                if not np.all(train):
                    incorrect_indx = np.where(train == 0)[0]
                    i_labels = ''
                    for idx in incorrect_indx:
                        i_labels += f' {labs[idx]} (gt: {ds.get_ds_classes()[labels.reshape((-1, 8))[t_number][idx]]},' \
                                    f' assigned: {ds.get_ds_classes()[symbolic.reshape((-1, 8))[t_number][idx]]})'
                    debug_text += f'car {t_number + 1}:{i_labels}. \n'

            if debug:
                plot_mask(output[0], i, ds.get_pil_image(i), tag='prediction')
        train_accuracies.append(accuracy)
        # debug_text = f"image {i}/{samples}, accuracy score: {round(accuracy * 100, 1)}%, " \
        #              f"running accuracy score: {(np.mean(train_accuracies) * 100).round(3)}%, " \
        #              f"Number of gt attributes {len(labels[labels > 0])}. "

        prog_bar.set_description(desc=f"Acc: {(np.mean(train_accuracies) * 100).round(3)}")
        if debug and accuracy != 1:
            print(debug_text)
            # print(issues)
            print(symbolic)
            print(labels)
        rtpt.step()

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
