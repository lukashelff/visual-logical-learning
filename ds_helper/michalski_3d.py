import json
import os
import random

import jsonpickle
import torch
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from blender_image_generator.json_util import merge_json_files, combine_json


class MichalskiTrainDataset(Dataset):
    def __init__(self, class_rule, base_scene, raw_trains, train_vis, ds_size=10000, resize=False, label_noise=0,
                 ds_path='output/image_generator', y_val='direction'
                 ):
        """ MichalskiTrainDataset
            Args:
                class_rule: string, rule for classifying the train either 'theoryx', 'numerical' or 'complex'
                base_scene: string, background scene of the train either 'base_scene', 'desert_scene', 'sky_scene' or 'fisheye_scene'
                raw_trains: string, type of train description either 'RandomTrains' or 'MichalskiTrains'
                train_vis: string, visualization of the train description either 'MichalskiTrains' or 'SimpleObjects'
                ds_size: int, number of ds images
                resize: bool if true images are resized to 224x224
                label_noise: float, noise added to the labels
                ds_path: string, path to the ds
                y_val: string, label to be predicted either 'direction' or 'attribute'
            @return:
                X_val: X value output for training data returned in __getitem__()
                ['image', 'predicted_attributes', 'gt_attributes', 'gt_attributes_individual_class', 'predicted_mask', gt_mask]
                        image (torch): image of michalski train

                y_val: ['direction','attribute','mask'] y label output for training data returned in __getitem__()

            """
        # ds data
        self.images = []
        self.trains = []
        self.masks = []
        # ds settings
        self.class_rule, self.base_scene, self.raw_trains, self.train_vis = class_rule, base_scene, raw_trains, train_vis
        self.resize, self.train_count = resize, ds_size
        self.label_noise = label_noise
        self.y_val = y_val

        # ds path
        ds_typ = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}'
        self.image_base_path = f'{ds_path}/{ds_typ}/images'
        self.all_scenes_path = f'{ds_path}/{ds_typ}/all_scenes'

        # ds labels
        self.labels = ['direction']
        self.label_classes = ['west', 'east']
        self.attributes = ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2', 'load_obj3'] * 4
        color = ['yellow', 'green', 'grey', 'red', 'blue']
        length = ['short', 'long']
        walls = ["braced_wall", 'solid_wall']
        roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
        wheel_count = ['2_wheels', '3_wheels']
        load_obj = ['diamond', "box", "golden_vase", 'barrel', 'metal_pot', 'oval_vase']
        self.attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
        if y_val == 'direction':
            self.class_dim = len(self.label_classes)
            self.output_dim = len(self.labels)
        elif y_val == 'attribute':
            self.class_dim = len(self.attribute_classes)
            self.output_dim = len(self.attributes)

        # check ds consistency
        if not os.path.isfile(self.all_scenes_path + '/all_scenes.json'):
            raise AssertionError('json scene file missing. Not all images were generated')
        if len(os.listdir(self.image_base_path)) < self.train_count:
            raise AssertionError(f'Missing images in dataset. Expected size {self.train_count}.'
                                 f'Available images: {len(os.listdir(self.image_base_path))}')

        # load ds
        path = self.all_scenes_path + '/all_scenes.json'
        with open(path, 'r') as f:
            all_scenes = json.load(f)
            for scene in all_scenes['scenes'][:ds_size]:
                self.images.append(scene['image_filename'])
                # self.depths.append(scene['depth_map_filename'])
                train = jsonpickle.decode(scene['m_train'])
                self.trains.append(train)
                self.masks.append(scene['car_masks'])
        # init ds transforms
        trans = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ]
        if resize:
            print('resize true')
            trans.append(transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC))
        self.norm = transforms.Compose(trans)
        self.normalize_mask = transforms.Compose([
            # transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5]),
        ])
        # add noise to labels
        if label_noise > 0:
            print(f'applying noise of {label_noise} to dataset labels')
            for train in self.trains:
                n = random.random()
                if n < label_noise:
                    lab = train.get_label()
                    if lab == 'east':
                        train.set_label('west')
                    elif lab == 'west':
                        train.set_label('east')
                    else:
                        raise ValueError(f'unexpected label value {lab}, expected value east or west')

    def __getitem__(self, item):
        image = self.get_pil_image(item)
        X = self.norm(image)
        if self.y_val == 'direction':
            y = self.get_direction(item)
        elif self.y_val == 'attribute':
            y = self.get_attributes(item)
        else:
            raise ValueError(f'unknown y_val {self.y_val}')
        return X, y

    def __len__(self):
        return self.train_count

    def get_direction(self, item):
        lab = self.trains[item].get_label()
        if lab == 'none':
            # return torch.tensor(0).unsqueeze(dim=0)
            raise AssertionError(f'There is no direction label for a RandomTrains. Use MichalskiTrain DS.')
        label_binary = self.label_classes.index(lab)
        label = torch.tensor(label_binary).unsqueeze(dim=0)
        return label

    def get_attributes(self, item):
        att = self.attribute_classes
        train = self.trains[item]
        cars = train.get_cars()
        labels = [0] * 32
        # each train has (4 cars a 8 attributes) totalling to 32 labels
        # each label can have 22 classes
        for car in cars:
            # index 0 = not existent
            color = att.index(car.get_blender_car_color())
            length = att.index(car.get_car_length())
            wall = att.index(car.get_blender_wall())
            roof = att.index(car.get_blender_roof())
            wheels = att.index(car.get_car_wheels())
            l_shape = att.index(car.get_blender_payload())
            l_num = car.get_load_number()
            l_shapes = [l_shape] * l_num + [0] * (3 - l_num)
            car_number = car.get_car_number()
            labels[8 * (car_number - 1):8 * car_number] = [color, length, wall, roof, wheels] + l_shapes
        return torch.tensor(labels)

    def get_m_train(self, item):
        return self.trains[item]

    def get_mask(self, item):
        return self.masks[item]

    def get_pil_image(self, item):
        im_path = self.get_image_path(item)
        return Image.open(im_path).convert('RGB')

    def get_image_path(self, item):
        return self.image_base_path + '/' + self.images[item]

    def get_label_for_id(self, item):
        return self.trains[item].get_label()

    def get_trains(self):
        return self.trains

    def get_ds_labels(self):
        return self.labels


def get_datasets(base_scene, raw_trains, train_vis, ds_size, ds_path, y_val='direction', class_rule='theoryx',
                 resize=False, noise=0):
    """
    Load dataset from file if they exist.

    :param base_scene: base scene name
    :param raw_trains: train description
    :param train_vis: train visualization
    :param ds_size: size of the dataset
    :param ds_path: path to the dataset folder
    :param y_val: value to predict, either direction or attribute
    :param class_rule: class rule to use
    :param resize: resize images to 224x224
    :param noise: noise to apply to labels
    :return: Michalski-3D dataset
    """
    path_ori = f'{ds_path}/{train_vis}_{class_rule}_{raw_trains}_{base_scene}'
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        combine_json(base_scene, raw_trains, train_vis, class_rule, out_dir=path_ori, ds_size=ds_size)
        raise Warning(f'Dataloader did not find JSON ground truth information.'
                      f'Might be caused by interruptions during process of image generation.'
                      f'Generating new JSON file at: {path_ori + "/all_scenes/all_scenes.json"}')
    im_path = path_ori + '/images'
    if not os.path.isdir(im_path):
        raise AssertionError('dataset not found, please generate images first')

    files = os.listdir(im_path)
    # total image count equals 10.000 adjust if not all images need to be generated
    if len(files) < ds_size:
        raise AssertionError(
            f'not enough images in dataset: expected {ds_size}, present: {len(files)}'
            f' please generate the missing images')
    elif len(files) > ds_size:
        raise Warning(
            f' dataloader did not select all images of the dataset, number of selected images:  {ds_size},'
            f' available images in dataset: {len(files)}')

    # merge json files to one if it does not already exist
    if not os.path.isfile(path_ori + '/all_scenes/all_scenes.json'):
        raise AssertionError(
            f'no JSON found')
    # image_count = None for standard image count
    full_ds = MichalskiTrainDataset(class_rule=class_rule, base_scene=base_scene, raw_trains=raw_trains,
                                    train_vis=train_vis, label_noise=noise, y_val=y_val,
                                    ds_size=ds_size, resize=resize, ds_path=ds_path)
    return full_ds
