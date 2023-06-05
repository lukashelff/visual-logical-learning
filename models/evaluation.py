import json
import os
from itertools import product
from random import sample

import jsonpickle
import pandas as pd
import torch
from rtpt import RTPT
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
    tr_size = 1000
    inference_size = 2000
    trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path,
                      setup_model=False, setup_ds=False)
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


def intervention_rcnn(args):
    from models.trainer import Trainer
    batch_size = 20
    device = torch.device("cpu" if not torch.cuda.is_available() or args.cuda == -1 else f"cuda")

    trainer = Trainer(args.background, args.description, args.visualization, device, args.model, args.rule,
                      args.ds_path + '/intervention', ds_size=12,
                      y_val=args.y_val, resume=True, batch_size=batch_size, setup_model=False, setup_ds=True,
                      min_car=args.min_train_length, max_car=args.max_train_length, val_size=12, train_size=None)
    model_path = 'output/models/multi_label_rcnn/maskv2_classification/Trains_theoryx_RandomTrains_base_scene/imcount_12000_X_val_image_pretrained_lr_0.001_step_10000_gamma0.1/'
    trainer.setup_model(True, path=model_path, y_val=args.y_val)
    from models.rcnn.plot_prediction import predict_and_plot
    from models.rcnn.inference import infer_symbolic

    infer_symbolic(trainer.model, trainer.dl['val'], device, debug=True)


def ood(device, ds_path):
    from models.trainer import Trainer
    base_scene, raw_trains, train_vis = 'base_scene', 'MichalskiTrains', 'Trains'
    train_type = ['MichalskiTrains', 'RandomTrains'][1]
    # device = 'cpu'
    rules = ['theoryx', 'numerical', 'complex'][:1]
    tr_sizes = [100, 1000, 10000]
    models = ['resnet18', 'EfficientNet', 'VisionTransformer']
    inference_size = 2000
    val_ids = sample(range(12000), inference_size)
    # val_ids = [i for i in range(2000)]

    neur_data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'image noise',
                 'label noise', 'Validation acc', 'Validation precision', 'Validation recall'])
    rtpt = RTPT(name_initials='LH', experiment_name='odd', max_iterations=len(rules) * len(models) * len(tr_sizes) * 5)
    rtpt.start()
    for class_rule, model_name, tr_size, cv in product(rules, models,
                                                       tr_sizes, range(5)):
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path,
                          setup_model=False, setup_ds=False)
        model_path = trainer.get_model_path(prefix=True, im_count=tr_size, suffix=f'it_{cv}/', model_name=model_name)
        trainer.setup_model(True, path=model_path, y_val='direction')
        path = f'{ds_path}/{train_vis}_{class_rule}_{train_type}_{base_scene}_len_2-4'
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

        TP, FP, TN, FN = 0, 0, 0, 0
        for idx in tqdm(val_ids):
            image, lab = images[idx], labels[idx]
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

            elif pred == 'east' and lab == 'west':
                FP += 1
                cat = 'FP'

        acc = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
        precision = round(TP / (TP + FP) * 100, 2)
        recall = round(TP / (TP + FN) * 100, 2)
        df = pd.DataFrame(
            [[model_name, tr_size, class_rule, train_vis, base_scene, cv, 0, 0, acc, precision, recall]],
            columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'image noise',
                     'label noise', 'Validation acc', 'Validation precision', 'Validation recall'])
        neur_data = pd.concat([neur_data, df])
        print(
            f'model: {model_name}, class_rule: {class_rule}, training samples {tr_size}, cv: {cv}, TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN},'
            f' acc: {acc}%, precision: {precision}%, recall: {recall}%', flush=True)
        del trainer
        rtpt.step()
    os.makedirs('output/ood', exist_ok=True)
    neur_data.to_csv(f'output/neural/ood/ood_{train_vis}_{train_type}_{base_scene}_len_2-4.csv')
