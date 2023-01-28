import glob
import json
import os

import numpy as np
import pandas as pd


def get_cv_data(out_path, y_val):
    data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'label', 'epoch',
                 'Validation acc', 'noise'])
    data_acum_acc = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'epoch',
                 'Validation acc', 'Train acc', 'noise'])
    data_ev = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'mean', 'variance', 'std', 'noise'])
    models = os.listdir('output/models/')
    if 'old' in models:
        models.remove('old')
    for model_name in models:

        path1 = f'output/models/{model_name}/{y_val}_classification/'
        try:
            datasets = os.listdir(path1)
        except:
            datasets = []
        visualizations = set([ds.split('_')[0] for ds in datasets])
        rules = set([ds.split('_')[1] for ds in datasets])
        train_typs = set([ds.split('_')[2] for ds in datasets])
        scenes = set([ds.split('_')[3] + '_' + ds[4] for ds in datasets])
        for ds in datasets:
            sets = ds.split('_')
            visualization = sets[0]
            rule = sets[1]
            train_typ = sets[2]
            scene = sets[3] + '_' + sets[4]
            path2 = path1 + f'{ds}/'
            noises = os.listdir(path2)
            for noise in noises:
                path3 = path2 + f'{noise}/'
                ns = 0 if len(noise) == 2 else float(noise.split('_')[1][:3])
                configs = os.listdir(path3)
                configs.insert(0, configs.pop())
                for config in configs:

                    conf = config.split('_')
                    imcount = int(conf[1])
                    cv_paths = glob.glob(path3 + config + '/*/metrics.json')
                    final_acc = []
                    for iteration, path in enumerate(cv_paths):
                        with open(path, 'r') as fp:
                            statistics = json.load(fp)
                            epoch_label_accs = statistics['epoch_label_accs']['val']
                            epoch_acum_accs = statistics['epoch_acum_accs']
                            epoch_loss = statistics['epoch_loss']['val']
                            num_epochs = len(epoch_loss)
                            final_acc.append(epoch_acum_accs['val']['acc'][-1])
                            labels = [key for key in epoch_label_accs][:-2]
                            for label in labels:
                                val_acc = epoch_label_accs[label]
                                li = []
                                for epoch in range(num_epochs):
                                    acc = val_acc['acc'][epoch]
                                    li.append(
                                        [model_name, imcount, rule, visualization, scene, iteration, label, epoch, acc,
                                         ns])
                                _df = pd.DataFrame(li, columns=['Methods', 'number of images', 'rule', 'visualization',
                                                                'scene',
                                                                'cv iteration', 'label', 'epoch', 'Validation acc',
                                                                'noise'])
                                data = pd.concat([data, _df], ignore_index=True)
                                li = []
                                for epoch in range(num_epochs):
                                    acc = epoch_acum_accs['val']['acc'][epoch]
                                    acc_train = epoch_acum_accs['train']['acc'][epoch]
                                    li.append(
                                        [model_name, imcount, rule, visualization, scene, iteration, epoch, acc,
                                         acc_train, ns])
                                _df = pd.DataFrame(li, columns=['Methods', 'number of images', 'rule', 'visualization',
                                                                'scene',
                                                                'cv iteration', 'epoch', 'Validation acc', 'Train acc',
                                                                'noise'])
                                data_acum_acc = pd.concat([data_acum_acc, _df], ignore_index=True)
                    if len(final_acc) > 0:
                        final_acc = np.array(final_acc) * 100
                        mean = sum(final_acc) / len(final_acc)
                        variance = sum((xi - mean) ** 2 for xi in final_acc) / len(final_acc)
                        std = np.sqrt(variance)
                        li = [model_name, imcount, rule, visualization, scene, mean, variance, std, ns]
                        _df = pd.DataFrame([li],
                                           columns=['Methods', 'number of images', 'rule', 'visualization', 'scene',
                                                    'mean',
                                                    'variance', 'std', 'noise'])
                        data_ev = pd.concat([data_ev, _df], ignore_index=True)
    os.makedirs(out_path, exist_ok=True)
    data.to_csv(out_path + 'label_acc_over_epoch.csv')
    data_acum_acc.to_csv(out_path + 'mean_acc_over_epoch.csv')
    data_ev.to_csv(out_path + 'mean_variance_comparison.csv')