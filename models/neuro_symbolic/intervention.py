import glob
import os
from itertools import product
import torch

import pandas as pd

from raw.concept_tester import eval_rule


def intervention_neuro_symbolic(ns_pth, device):
    ns_stats_path = f'{ns_pth}/stats'
    interventions = ['intervention1', 'intervention1b', 'intervention2', 'intervention2b']
    ilps = ['popper', 'aleph']
    dirs = glob.glob(ns_stats_path + '/*.csv')
    ilp_data = []
    for dir in dirs:
        if 'Trains' in dir:
            with open(dir, 'r') as f:
                ilp_data.append(pd.read_csv(f))
    ilp_data = pd.concat(ilp_data, ignore_index=True)
    ilp_data['Train length'] = '2-4'
    ilp_data = ilp_data.loc[ilp_data['noise'] == 0]

    ilp_models = sorted(ilp_data['Methods'].unique())
    t_sizes = [1000]
    t_sizes = [100, 1000, 10000]
    rule = 'theoryx'
    min_car, max_car = 2, 4
    base_scene, raw_trains, train_vis = 'base_scene', 'MichalskiTrains', 'Trains'

    stats = pd.DataFrame(columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration',
                                  'label', 'intervention', 'Validation acc', "precision", "recall"])
    for model_name, intervention, cv, tr_sample in product(ilps,
                                                           interventions, [*range(5)], [100, 1000, 10000]):
        f = ilp_data.loc[ilp_data['Methods'] == model_name].loc[ilp_data['training samples'] == tr_sample].loc[
            ilp_data['rule'] == rule].loc[ilp_data['cv iteration'] == cv]
        if f.empty:
            print(f'No data for {model_name}, {rule}, {cv}, {tr_sample}')
            continue
        theory = f.iloc[0]['theory']
        val_path = f'output/models/multi_label_rcnn/inferred_ds/prediction/{intervention}/Trains_MichalskiTrains_len_{min_car}-{max_car}.txt'
        if not os.path.exists(val_path):
            ds_path = 'TrainGenerator/output/image_generator'
            infer_ds(intervention, base_scene, raw_trains, train_vis, device, ds_path, 2000, rule, min_car, max_car)
        TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = eval_rule(theory=theory, ds_val=val_path,
                                                                           ds_train=None, dir='TrainGenerator/',
                                                                           print_stats=False, )
        acc = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
        precision = 0 if TP + FP == 0 else round(TP / (TP + FP) * 100, 2)
        recall = 0 if TP + FN == 0 else round(TP / (TP + FN) * 100, 2)
        frame = [
            [model_name, tr_sample, rule, train_vis, base_scene, cv, 'direction', intervention, acc, precision,
             recall]]
        _df = pd.DataFrame(frame,
                           columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration',
                                    'label', 'intervention', 'Validation acc', "precision", "recall"])
        stats = pd.concat([stats, _df], ignore_index=True)
    o_path = f'output/model_comparison/interventions/neuro_symbolic_intervention.csv'
    os.makedirs(os.path.dirname(o_path), exist_ok=True)
    stats.to_csv(o_path)


def infer_ds(intervention, base_scene, raw_trains, train_vis, device, ds_path, ds_size, class_rule, min_cars, max_cars):
    from models.trainer import Trainer
    batch_size = 10 if torch.cuda.get_device_properties(0).total_memory > 9000000000 else 1
    num_epochs = 30
    train_size, val_size = 12000, 2000
    # every n training steps, the learning rate is reduced by gamma
    num_batches = (train_size * num_epochs) // batch_size
    step_size = num_batches // 3
    lr = 0.001
    gamma = 0.1
    v2 = 'v2'
    y_val = f'mask{v2}'
    trainer = Trainer(base_scene, raw_trains, train_vis, device, 'multi_label_rcnn', intervention, ds_path,
                      ds_size=ds_size, lr=lr, step_size=step_size, gamma=gamma, min_car=min_cars, max_car=max_cars,
                      y_val=y_val, resume=True, batch_size=batch_size, setup_model=False, setup_ds=False)
    model_path = f"output/models/multi_label_rcnn/mask{v2}_classification/{train_vis}_theoryx_RandomTrains_base_scene/imcount_12000_X_val_image_pretrained_lr_0.001_step_10000_gamma0.1/"
    trainer.setup_ds(val_size=ds_size)
    trainer.setup_model(resume=True, path=model_path)
    out_path = f'output/models/multi_label_rcnn/inferred_ds/{intervention}'
    from models.rcnn.inference import infer_dataset
    infer_dataset(trainer.model, trainer.dl['val'], device, out_path, train_vis, class_rule, 'MichalskiTrains',
                  min_cars, max_cars)
