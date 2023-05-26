import json
import os
from itertools import product

import jsonpickle
import pandas as pd
from tqdm import tqdm

from models.trainer import Trainer


def generalization_test(min_cars, max_cars, base_scene, raw_trains, train_vis, device, ds_path, ds_size=None):
    ds_size = ds_size if ds_size is not None else 2000
    data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'label',
                 'Validation acc', "precision", "recall"])
    for model, rule, tr_samples in product(['resnet18', 'EfficientNet', 'VisionTransformer'],
                                           ['theoryx', 'numerical', 'complex'], [100, 1000, 10000]):
        model_name = model
        resize = False
        batch_size = 25
        lr = 0.001
        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001

        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, rule, ds_path,
                          ds_size=ds_size, setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize,
                          lr=lr, resume=True, min_car=min_cars, max_car=max_cars)
        for cv in range(5):
            pth = trainer.get_model_path(prefix=True, im_count=tr_samples, suffix=f'it_{cv}/', model_name=model_name)
            acc, precision, recall = trainer.val(model_path=pth)
            frame = [[model_name, tr_samples, rule, train_vis, base_scene, cv, 'direction', acc, precision, recall]]
            _df = pd.DataFrame(frame, columns=['Methods', 'number of images', 'rule', 'visualization', 'scene',
                                               'cv iteration', 'label', 'Validation acc', "precision", "recall"])
            data = pd.concat([data, _df], ignore_index=True)
    os.makedirs(f'output/model_comparison/generalization', exist_ok=True)
    data.to_csv(f'output/model_comparison/generalization/cnn_generalization_{min_cars}_{max_cars}.csv')


def zoom_test(min_cars, max_cars, base_scene, raw_trains, train_vis, device, ds_path, ds_size=None):
    ds_size = ds_size if ds_size is not None else 2000
    data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'label',
                 'Validation acc', "precision", "recall"])
    rules = ['theoryx', 'numerical', 'complex'][:1]
    tr_ims = [100, 1000, 10000]

    for model, rule, tr_samples in product(['resnet18', 'EfficientNet', 'VisionTransformer'],
                                           rules, tr_ims):
        model_name = model
        resize = False
        batch_size = 25
        lr = 0.001
        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, rule, ds_path,
                          ds_size=ds_size, setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize,
                          lr=lr, resume=True, min_car=min_cars, max_car=max_cars)
        for cv in range(5):
            pth = trainer.get_model_path(prefix=True, im_count=tr_samples, suffix=f'it_{cv}/', model_name=model_name)
            acc, precision, recall = trainer.val(model_path=pth, val_size=2000)
            frame = [[model_name, tr_samples, rule, train_vis, base_scene, cv, 'direction', acc, precision, recall]]
            _df = pd.DataFrame(frame, columns=['Methods', 'number of images', 'rule', 'visualization', 'scene',
                                               'cv iteration', 'label', 'Validation acc', "precision", "recall"])
            data = pd.concat([data, _df], ignore_index=True)
    os.makedirs(f'output/model_comparison/generalization', exist_ok=True)
    data.to_csv(f'output/model_comparison/generalization/cnn_zoom_{min_cars}_{max_cars}.csv')


def intervention_test(model_name, device, ds_path):
    from models.trainer import Trainer
    base_scene, raw_trains, train_vis, class_rule = 'base_scene', 'MichalskiTrains', 'Trains', 'theoryx'
    train_type = ['MichalskiTrains', 'RandomTrains'][0]
    ds_iv = ['intervention/', ''][0]
    # model_name = 'EfficientNet'
    # device = 'cpu'
    resize = True if model_name == 'VisionTransformer' else False
    tr_size = 1000
    inference_size = 2000
    trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path,
                      setup_model=False, setup_ds=False, resize=resize)
    model_path = trainer.get_model_path(prefix=True, im_count=tr_size, suffix=f'it_0/', model_name=model_name)
    trainer.setup_model(True, path=model_path, y_val='direction')
    path = f'{ds_path}/{ds_iv}{train_vis}_{class_rule}_{train_type}_{base_scene}_len_2-4'
    scene_path = f'{path}/all_scenes/all_scenes.json'
    with open(scene_path, 'r') as f:
        # count number of files in dir
        n = len(os.listdir(f'{path}/images'))
        images = [None] * n
        labels = [None] * n
        all_scenes = json.load(f)
        for scene in all_scenes['scenes']:
            train = scene['m_train']
            train = jsonpickle.decode(train)
            labels[scene['image_index']] = train.get_label()
            images[scene['image_index']] = scene['image_filename']

    images, labels = images[:inference_size], labels[:inference_size]
    TP, FP, TN, FN = 0, 0, 0, 0
    for image, lab in tqdm(zip(images, labels)):
        im_path = f'{path}/images/{image}'
        pred = trainer.infer_im(im_path)
        pred = 'west' if pred[0] == 0 else 'east'
        if pred == 'west' and lab == 'west':
            TN += 1
            cat = 'TN'
        elif pred == 'east' and lab == 'east':
            TP += 1
            cat = 'TP'
        elif pred == 'west' and lab == 'east':
            FN += 1
            cat = 'FN'
            # print(f'{cat} image {os.path.basename(im_path)} (pred: {pred} ,gt: {lab})')

        elif pred == 'east' and lab == 'west':
            FP += 1
            cat = 'FP'
        if pred != lab or len(labels) < 100:
            print(f'{cat} image {os.path.basename(im_path)} (pred: {pred} ,gt: {lab})')
    acc = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
    precision = round(TP / (TP + FP) * 100, 2)
    recall = round(TP / (TP + FN) * 100, 2)
    print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}, acc: {acc}%, precision: {precision}%, recall: {recall}%',
          flush=True)
