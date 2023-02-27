import glob
from itertools import product
import pandas as pd
from models.trainer import Trainer
from raw.concept_tester import eval_rule


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
    data.to_csv(f'output/model_comparison/generalization/cnn_generalization_{min_cars}_{max_cars}.csv')


def ilp_generalization_test(ilp_pth, min_cars, max_cars):
    ilp_stats_path = f'{ilp_pth}/stats'

    dirs = glob.glob(ilp_stats_path + '/*.csv')
    ilp_data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            ilp_data.append(pd.read_csv(f))
    ilp_data = pd.concat(ilp_data, ignore_index=True)
    ilp_data['Train length'] = '2-4'
    ilp_data = ilp_data.loc[ilp_data['noise'] == 0]
    ilp_models = sorted(ilp_data['Methods'].unique())

    data = pd.DataFrame(
        columns=['Methods', 'training samples', 'rule', 'cv iteration', 'label', 'Validation acc', "precision",
                 "recall"])
    for model_name, rule, cv, tr_sample in product(['popper', 'aleph'], ['theoryx', 'numerical', 'complex'],
                                                   [*range(5)], [100, 1000, 10000]):
        theory = ilp_data.loc[ilp_data['Methods'] == model_name].loc[ilp_data['rule'] == rule].loc[
            ilp_data['cv iteration'] == cv].iloc[0]['theory'].loc[ilp_data['training samples'] == tr_sample]
        val_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{rule}/MichalskiTrains_car_length_{min_cars}-{max_cars}.txt'
        TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = eval_rule(theory=theory, ds_val=val_path,
                                                                           ds_train=None, dir='TrainGenerator/',
                                                                           print_stats=False, )
        acc = (TP + TN) / (TP + TN + FP + FN)
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        frame = [[model_name, tr_sample, rule, cv, 'direction', acc, precision, recall]]
        _df = pd.DataFrame(frame, columns=['Methods', 'training samples', 'rule',
                                           'cv iteration', 'label', 'Validation acc', "precision", "recall"])
        data = pd.concat([data, _df], ignore_index=True)
    data.to_csv(f'output/model_comparison/generalization/ilp_generalization_{min_cars}_{max_cars}.csv')
