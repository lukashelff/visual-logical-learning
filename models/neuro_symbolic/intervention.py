import glob
import os
from itertools import product
import torch

import pandas as pd

from raw.concept_tester import eval_rule
from visualization.data_handler import get_ilp_neural_data


def intervention_neuro_symbolic(outpath, device):
    interventions = ['intervention1', 'intervention1b', 'intervention2', 'intervention2b']
    # ilps = ['popper', 'aleph']
    ilp_stats_path = f'{outpath}/ilp/stats'
    neuro_sym_path = f'{outpath}/neuro-symbolic/stats'
    alpha_ilp = f'{outpath}/neuro-symbolic/alphailp/stats'

    # dirs = glob.glob(neuro_sym_path + '/Trains*.csv')
    # ilp_data = []
    # for dir in dirs:
    #     if 'Trains' in dir:
    #         with open(dir, 'r') as f:
    #             ilp_data.append(pd.read_csv(f))
    # ilp_data = pd.concat(ilp_data, ignore_index=True)
    # ilp_data['Train length'] = '2-4'

    data, _, _, _ = get_ilp_neural_data(None, None, neuro_sym_path, None)
    data = data.loc[data['image noise'] == 0].loc[data['label noise'] == 0]

    models = sorted(data['Models'].unique()) + ['αILP']
    # models = ['αILP']
    t_sizes = [1000]
    # t_sizes = [100, 1000, 10000]
    rule = 'theoryx'
    min_car, max_car = 2, 4
    base_scene, raw_trains, train_vis = 'base_scene', 'MichalskiTrains', 'Trains'

    stats = pd.DataFrame(columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration',
                                  'label', 'intervention', 'Validation acc', "precision", "recall"])
    for model_name, intervention, cv, tr_sample in product(models, interventions, [*range(5)], t_sizes):
        if model_name == 'αILP':
            pth = f'output/neuro-symbolic/alphailp/intervention/converted/fold_{cv}.csv'
            with open(pth, 'r') as f:
                theory = f.read()
        else:
            f = data.loc[data['Models'] == model_name].loc[data['training samples'] == tr_sample].loc[
                data['rule'] == rule].loc[data['cv iteration'] == cv]
            if f.empty:
                print(f'No data for {model_name}, {rule}, {cv}, {tr_sample}, {intervention}')
                continue
            theory = f.iloc[0]['theory']

        val_path = f'output/models/multi_label_rcnn/inferred_ds/{intervention}/prediction/{rule}/{train_vis}_{raw_trains}_len_{min_car}-{max_car}.txt'
        if not os.path.exists(val_path):
            print(F'No data found for {val_path}')
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
    stats = stats.loc[stats['number of images'] == 1000]
    for model in models:
        # model out print 'RCNN-Aleph &  $99.83 \pm 0.38 \rightarrow 98.01 $\pm$ 3.29$  &  $99.82 $\pm$ 0.13 \rightarrow 100.0 $\pm$ 0.0$'
        model_p = model if model != 'αILP' else '$\\alpha$ILP'
        for intervention in interventions:
            # evaluate ground truth rule
            # rule_pth = 'TrainGenerator/example_rules/theoryx_rule.pl'
            # val_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{intervention}/{raw_trains}_len_{min_car}-{max_car}.txt'
            # TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = eval_rule(theory=rule_pth, ds_val=val_path,
            #                                                                    ds_train=None, dir='TrainGenerator/',
            #                                                                    print_stats=False, )
            # gt_acc = round((TP + TN) / (TP + TN + FP + FN) * 100, 2)
            model_stats = stats[(stats['Methods'] == model) & (stats['intervention'] == intervention)]
            # print mean and std of acc
            mean_acc = round(model_stats['Validation acc'].mean(), 2)
            std_acc = round(model_stats['Validation acc'].std(), 2)
            if intervention == 'intervention1' or intervention == 'intervention2':
                model_p += f"& ${mean_acc} \\pm {std_acc}  \\rightarrow "
            else:
                model_p += f'{mean_acc} \pm {std_acc} $ '
            if intervention == 'intervention2b':
                model_p += '\\\\ \hline'
        print(model_p)


def convert_theory(theory):
    # input predicates:
    # kp: 1:train
    # in: 2:car, train
    # car_num: 2:car, int
    # color: 2:car, color
    # length: 2:car, length
    # wall: 2:car, wall
    # roof: 2:car, roof
    # wheel: 2:car, int
    # load1: 2:car, load
    # load2: 2:car, load
    # load3: 2:car, load
    # output predicates:
    # eastbound: 1:train
    # has_car: 2:train, car
    # car_num: 2:car, int
    # car_color: 2:car, color
    # car_length: 2:car, length
    # car_wall: 2:car, wall
    # has_roof2: 2:car, roof
    # has_wheel0: 2:car, int
    # has_payload: 2:car, load
    # load_num: 2:car, int

    # convert input predicates
    theory = theory.replace('kp', 'eastbound')
    theory = theory.replace('in', 'has_car')

    return theory


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
    trainer.setup_ds(train_size=1, val_size=ds_size - 1)
    trainer.setup_model(resume=True, path=model_path)
    out_path = f'output/models/multi_label_rcnn/inferred_ds/{intervention}'
    from models.rcnn.inference import infer_dataset
    infer_dataset(trainer.model, trainer.dl['val'], device, out_path, train_vis, class_rule, 'MichalskiTrains',
                  min_cars, max_cars)
