import glob
import json
import os

import numpy as np
import pandas as pd


def get_cv_data(out_path, y_val, models=['EfficientNet', 'resnet18', 'VisionTransformer']):
    data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'label', 'epoch',
                 'Validation acc', 'noise', 'noise type'])
    data_acum_acc = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'epoch',
                 'Validation acc', 'Train acc', 'noise', 'noise type'])
    data_ev = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'mean', 'variance', 'std', 'noise',
                 'noise type'])
    # models = os.listdir('output/models/')
    # if 'old' in models:
    #     models.remove('old')
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
                if 'im_noise' in noise:
                    noise_type = 'image noise'
                elif 'noise' in noise:
                    noise_type = 'label noise'
                else:
                    noise_type = 'no noise'
                configs = os.listdir(path3)
                configs.insert(0, configs.pop())
                for config in configs:
                    conf = config.split('_')
                    try:
                        imcount = int(conf[1])
                    except:
                        print('default')
                    cv_paths = glob.glob(path3 + config + '/*/metrics.json')
                    final_acc = []
                    for iteration, path in enumerate(cv_paths):
                        with open(path, 'r') as fp:
                            statistics = json.load(fp)
                            epoch_label_accs = statistics['epoch_label_accs']['val']
                            epoch_acum_accs = statistics['epoch_acum_accs']
                            epoch_loss = statistics['epoch_loss']['val']
                            num_epochs = len(epoch_loss)
                            # final_acc.append(max(epoch_acum_accs['val']['acc']))
                            final_acc.append(epoch_acum_accs['val']['acc'][-1])
                            labels = [key for key in epoch_label_accs][:-2]
                            for label in labels:
                                val_acc = epoch_label_accs[label]
                                li = []
                                for epoch in range(num_epochs):
                                    acc = val_acc['acc'][epoch]
                                    li.append(
                                        [model_name, imcount, rule, visualization, scene, iteration, label, epoch, acc,
                                         ns, noise_type])
                                _df = pd.DataFrame(li, columns=['Methods', 'number of images', 'rule', 'visualization',
                                                                'scene',
                                                                'cv iteration', 'label', 'epoch', 'Validation acc',
                                                                'noise', 'noise type'])
                                data = pd.concat([data, _df], ignore_index=True)
                                li = []
                                for epoch in range(num_epochs):
                                    acc = epoch_acum_accs['val']['acc'][epoch]
                                    acc_train = epoch_acum_accs['train']['acc'][epoch]
                                    li.append(
                                        [model_name, imcount, rule, visualization, scene, iteration, epoch, acc,
                                         acc_train, ns, noise_type])
                                _df = pd.DataFrame(li, columns=['Methods', 'number of images', 'rule', 'visualization',
                                                                'scene',
                                                                'cv iteration', 'epoch', 'Validation acc', 'Train acc',
                                                                'noise', 'noise type'])
                                data_acum_acc = pd.concat([data_acum_acc, _df], ignore_index=True)
                    if len(final_acc) > 0:
                        final_acc = np.array(final_acc) * 100
                        mean = sum(final_acc) / len(final_acc)
                        variance = sum((xi - mean) ** 2 for xi in final_acc) / len(final_acc)
                        std = np.sqrt(variance)
                        li = [model_name, imcount, rule, visualization, scene, mean, variance, std, ns, noise_type]
                        _df = pd.DataFrame([li],
                                           columns=['Methods', 'number of images', 'rule', 'visualization', 'scene',
                                                    'mean',
                                                    'variance', 'std', 'noise', 'noise type'])
                        data_ev = pd.concat([data_ev, _df], ignore_index=True)
    os.makedirs(out_path, exist_ok=True)
    data.to_csv(out_path + '/label_acc_over_epoch.csv')
    data_acum_acc.to_csv(out_path + '/mean_acc_over_epoch.csv')
    data_ev.to_csv(out_path + '/mean_variance_comparison.csv')


def get_ilp_neural_data(ilp_stats_path, neural_stats_path, neuro_symbolic_stats_path, vis='Train'):
    ilp_data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'epoch',
                 'Validation acc', 'noise'])
    ilp_models = []
    neuro_symbolic_data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'epoch',
                 'Validation acc', 'noise'])
    neuro_symbolic_models = []
    neur_data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'epoch',
                 'Validation acc', 'noise', 'noise type'])
    neural_models = []

    if ilp_stats_path is not None:
        dirs = glob.glob(ilp_stats_path + '/*.csv')
        ilp_data = []
        for dir in dirs:
            with open(dir, 'r') as f:
                ilp_data.append(pd.read_csv(f))
        ilp_data = pd.concat(ilp_data, ignore_index=True)
        ilp_data['visualization'] = 'Trains'
        ilp_data['noise type'] = 'label noise'
        ilp_data.loc[ilp_data['noise'] == 0, 'noise type'] = 'no noise'
        ilp_data['Methods'] = ilp_data['Methods'].apply(lambda x: x.title() + ' (Symb.)')
        ilp_models = sorted(ilp_data['Methods'].unique())


    if neuro_symbolic_stats_path is not None:
        dirs = glob.glob(neuro_symbolic_stats_path + '/*.csv')
        simple_objects_data, trains_data = [], []
        simple_objects_dirs, trains_dirs = [], []
        for dir in dirs:
            if 'SimpleObjects' in dir:
                simple_objects_dirs.append(dir)
            else:
                trains_dirs.append(dir)
        for dir in simple_objects_dirs:
            with open(dir, 'r') as f:
                simple_objects_data.append(pd.read_csv(f))
        simple_objects_data = pd.concat(simple_objects_data, ignore_index=True)
        simple_objects_data['visualization'] = 'SimpleObjects'

        for dir in trains_dirs:
            with open(dir, 'r') as f:
                trains_data.append(pd.read_csv(f))
        trains_data = pd.concat(trains_data, ignore_index=True)
        trains_data['visualization'] = 'Trains'
        neuro_symbolic_data = pd.concat([simple_objects_data, trains_data], ignore_index=True)
        neuro_symbolic_data['noise type'] = 'label noise'
        neuro_symbolic_data.loc[neuro_symbolic_data['noise'] == 0, 'noise type'] = 'no noise'
        neuro_symbolic_data['Methods'] = neuro_symbolic_data['Methods'].apply(lambda x: 'RCNN-' + x.title() + ' (NeSy)')
        neuro_symbolic_models = sorted(neuro_symbolic_data['Methods'].unique())

    if neural_stats_path is not None:
        with open(neural_stats_path, 'r') as f:
            neur_data = pd.read_csv(f)
            neur_data = neur_data.loc[neur_data['epoch'] == 24]
            if vis is not None and vis != 'all':
                neur_data = neur_data.loc[neur_data['visualization'] == vis]
        neur_data = neur_data.rename({'number of images': 'training samples'}, axis='columns')
        neur_data['Methods'] = neur_data['Methods'].apply(lambda x: x.replace('resnet18', 'ResNet18'))
        neural_models = sorted(neur_data['Methods'].unique())

    data = pd.concat([ilp_data, neur_data, neuro_symbolic_data], ignore_index=True)
    data['Validation acc'] = data['Validation acc'].apply(lambda x: x * 100)
    # replace 'simpleobjects' with 'block' in visualization column of data
    # replace 'trains' with 'michalski' in visualization column of data
    data['visualization'] = data['visualization'].apply(lambda x: x.replace('SimpleObjects', 'Block'))
    data['visualization'] = data['visualization'].apply(lambda x: x.replace('Trains', 'Michalski'))
    data.rename(columns={'Methods': 'Models'}, inplace=True)
    data.reset_index(drop=True, inplace=True)
    return data, ilp_models, neural_models, neuro_symbolic_models


def read_csv_stats(csv_path, train_length=7, noise=0, symb=True, vis='Michalski'):
    with open(csv_path, 'r') as f:
        data = pd.read_csv(f)
        data = data.rename({'number of images': 'training samples'}, axis='columns')
        data['Train length'] = train_length
        data['noise'] = noise
        data['Validation acc'] = data['Validation acc'].apply(lambda x: x * 100)
        data['Methods'] = data['Methods'].apply(lambda x: x.replace('resnet18', 'ResNet18'))
        data['Methods'] = data['Methods'].apply(lambda x: x.replace('simpleobjects', 'Block'))
        data['Methods'] = data['Methods'].apply(lambda x: x.replace('trains', 'Michalski'))
        if symb:
            data['Methods'] = data['Methods'].apply(lambda x: x.replace('popper', 'Popper (Symb.)'))
            data['Methods'] = data['Methods'].apply(lambda x: x.replace('aleph', 'Aleph (Symb.)'))
        else:
            data['Methods'] = data['Methods'].apply(lambda x: x.replace('popper', 'RCNN-Popper (NeSy)'))
            data['Methods'] = data['Methods'].apply(lambda x: x.replace('aleph', 'RCNN-Aleph (NeSy)'))
        data['visualization'] = vis
        data.rename(columns={'Methods': 'Models'}, inplace=True)

    return data
