import glob
import os
from itertools import product

import pandas as pd

from raw.concept_tester import eval_rule


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
        f = ilp_data.loc[ilp_data['Methods'] == model_name].loc[ilp_data['training samples'] == tr_sample].loc[
            ilp_data['rule'] == rule].loc[ilp_data['cv iteration'] == cv]
        if f.empty:
            continue
        theory = f.iloc[0]['theory']
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
    os.makedirs(f'output/model_comparison/generalization', exist_ok=True)
    data.to_csv(f'output/model_comparison/generalization/ilp_generalization_{min_cars}_{max_cars}.csv')


def intervention_symbolic_ilp(ilp_pth):
    ilp_stats_path = f'{ilp_pth}/stats'
    interventions = ['intervention1', 'intervention1b', 'intervention2', 'intervention2b']
    ilps = ['popper', 'aleph']
    dirs = glob.glob(ilp_stats_path + '/*.csv')
    ilp_data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            ilp_data.append(pd.read_csv(f))
    ilp_data = pd.concat(ilp_data, ignore_index=True)
    ilp_data['Train length'] = '2-4'
    ilp_data = ilp_data.loc[ilp_data['noise'] == 0]
    ilp_models = sorted(ilp_data['Methods'].unique())
    min_cars, max_cars = 2, 4
    rule = 'theoryx'
    base_scene, raw_trains, train_vis, class_rule = 'base_scene', 'MichalskiTrains', 'Trains', 'theoryx'
    stats = pd.DataFrame(columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration',
                                  'label', 'intervention', 'Validation acc', "precision", "recall"])
    for model_name, intervention, cv, tr_sample in product(ilps,
                                                           interventions, [*range(5)], [100, 1000, 10000]):
        f = ilp_data.loc[ilp_data['Methods'] == model_name].loc[ilp_data['training samples'] == tr_sample].loc[
            ilp_data['rule'] == rule].loc[ilp_data['cv iteration'] == cv]
        if f.empty:
            continue
        theory = f.iloc[0]['theory']
        val_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{intervention}/MichalskiTrains_len_{min_cars}-{max_cars}.txt'
        TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = eval_rule(theory=theory, ds_val=val_path,
                                                                           ds_train=None, dir='TrainGenerator/',
                                                                           print_stats=False, )
        acc = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
        precision = 0 if TP + FP == 0 else round(TP / (TP + FP) * 100, 2)
        recall = 0 if TP + FN == 0 else round(TP / (TP + FN) * 100, 2)
        frame = [
            [model_name, tr_sample, class_rule, train_vis, base_scene, cv, 'direction', intervention, acc, precision,
             recall]]
        _df = pd.DataFrame(frame,
                           columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration',
                                    'label', 'intervention', 'Validation acc', "precision", "recall"])
        stats = pd.concat([stats, _df], ignore_index=True)


    o_path = f'output/model_comparison/interventions/ilp_intervention_{min_cars}_{max_cars}.csv'
    os.makedirs(os.path.dirname(o_path), exist_ok=True)
    stats.to_csv(o_path)
    # filter for number of images = 1000
    stats = stats.loc[stats['number of images'] == 1000]
    for intervention in interventions:
        # evaluate ground truth rule
        rule_pth = 'TrainGenerator/example_rules/theoryx_rule.pl'
        val_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{intervention}/MichalskiTrains_len_{min_cars}-{max_cars}.txt'
        TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = eval_rule(theory=rule_pth, ds_val=val_path,
                                                                           ds_train=None, dir='TrainGenerator/',
                                                                           print_stats=False, )
        acc = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
        for model in ilps:
            model_stats = stats[(stats['Methods'] == model) & (stats['intervention'] == intervention)]
            # print mean and std of acc
            mean_acc = round(model_stats['Validation acc'].mean(), 2)
            std_acc = round(model_stats['Validation acc'].std(), 2)
            print(f'{model} {intervention} acc: {mean_acc} +/- {std_acc} (ground truth: {acc})')


