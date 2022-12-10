import glob
import importlib
import os
import random
from itertools import product

import pyswip
import seaborn as sns
import multiprocessing as mp
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

import matplotlib.lines as mlines
from pyswip import Prolog
from tabulate import tabulate

from ilp import visualization
from ilp.setup import create_bk

from sklearn.model_selection import StratifiedShuffleSplit

from raw.concept_tester import eval_rule
from rdm.wrappers import Aleph
from importlib import reload


class Ilp_trainer():

    def __int__(self, method):
        self.method = method

    # cmd = f"echo \"read_all(trains2/train). induce.\" | yap -s5000 -h20000 -l aleph.pl > log.txt 2>&1"
    def cross_val(self, raw_trains, rules=['numerical', 'theoryx', 'complex'], models=['aleph', 'popper'], folds=5,
                  train_count=[100, 1000, 10000], ds_size=12000, noise=0):

        ilp_stats_path = f'output/ilp/stats/'
        os.makedirs(ilp_stats_path, exist_ok=True)
        for model in models:
            for class_rule in rules:
                ds_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{raw_trains}_{class_rule}.txt'
                for train_size in train_count:
                    data = pd.DataFrame(
                        columns=['Methods', 'training samples', 'rule', 'cv iteration', 'Validation acc', 'theory',
                                 'noise'])
                    with open(ds_path, "r") as file:
                        all_data = file.readlines()
                        if len(all_data) != ds_size:
                            raise f'datasets of size {ds_size} however only {len(all_data)} datasamples were generated'
                        y = [l[0] for l in all_data]
                        sss = StratifiedShuffleSplit(n_splits=folds, train_size=train_size, test_size=2000,
                                                     random_state=0)
                        inputs = []
                        for fold, (tr_idx, val_idx) in enumerate(sss.split(np.zeros(len(y)), y)):
                            out_path = f'output/ilp/datasets/{raw_trains}_{class_rule}_{train_size}/cv_{fold}'
                            os.makedirs(out_path, exist_ok=True)
                            train_path = f'{out_path}/train_samples.txt'
                            val_path = f'{out_path}/val_samples.txt'
                            train_samples = map(all_data.__getitem__, tr_idx)
                            val_samples = map(all_data.__getitem__, val_idx)
                            for ds in [train_path, val_path]:
                                try:
                                    os.remove(ds)
                                except OSError:
                                    pass
                            with open(train_path, 'w+') as train, open(val_path, 'w+') as val:
                                train.writelines(train_samples)
                                val.writelines(val_samples)
                            create_bk(train_path, out_path, train_size, noise)
                            inputs.append(out_path)

                    if model == 'popper':
                        out = []
                        # for c, input in enumerate(inputs):
                        #     out.append(self.popper_train(input, print_stats=True))
                        with mp.Pool(5) as p:
                            out = p.map(self.popper_train, inputs)
                    elif model == 'aleph':
                        out = []
                        # for c, input in enumerate(inputs):
                        #     out.append(self.aleph_train(input, print_stats=True))
                        with mp.Pool(5) as p:
                            if noise > 0:
                                inputs = [(i, noise * train_size) for i in inputs]
                                out = p.starmap(self.aleph_train, inputs)
                            else:
                                out = p.map(self.aleph_train, inputs)
                    else:
                        raise ValueError(f'model: {model} not supported')
                    li = []
                    for o in out:
                        theory, TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = o
                        li.append([model, train_size, class_rule, fold,
                                   (TP + TN) / (TP + FN + TN + FP),
                                   (TP_train + TN_train) / (TP_train + FN_train + TN_train + FP_train), theory, noise])
                    _df = pd.DataFrame(li, columns=['Methods', 'training samples', 'rule', 'cv iteration',
                                                    'Validation acc', 'Train acc', 'theory', 'noise'])
                    data = pd.concat([data, _df], ignore_index=True)

                    csv = f'/{model}_{class_rule}_{train_size}smpl_{noise}noise.csv' if noise > 0 else f'/{model}_{class_rule}_{train_size}smpl.csv'
                    data.to_csv(ilp_stats_path + csv)

    def train(self, model, raw_trains, class_rule, train_size, val_size):
        ds_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{raw_trains}_{class_rule}.txt'
        out_path = f'output/ilp/datasets/{raw_trains}_{class_rule}_{train_size}/cv_-1'
        os.makedirs(out_path, exist_ok=True)
        train_path = f'{out_path}/train_samples.txt'
        val_path = f'{out_path}/val_samples.txt'
        # ds_size = 10
        noise = 0.0
        with open(ds_path, "r") as file:
            all_data = file.readlines()
            if len(all_data) < train_size:
                raise f'train of size of {train_size} selected however only {len(all_data)} datasamples were generated'
            train_samples = random.sample(all_data, train_size)
            val_samples = random.sample(all_data, val_size)
            with open(train_path, 'w+') as train, open(val_path, 'w+') as val:
                train.writelines(train_samples)
                val.writelines(val_samples)
            create_bk(train_path, out_path, train_size, noise)

            if model == 'popper':
                self.popper_train(out_path, print_stats=True)
            elif model == 'aleph':
                self.aleph_train(out_path, print_stats=True)
            else:
                raise ValueError(f'model: {model} not supported')

    def popper_train(self, path, print_stats=True):
        from popper.loop import learn_solution
        from popper.util import Settings, format_prog, order_prog, print_prog_score
        # # popper = reload(popper)
        # reload(popper.loop.learn_solution)
        popper_data = f'{path}/popper/gt1'
        train_path = f'{path}/train_samples.txt'
        val_path = f'{path}/val_samples.txt'
        settings = Settings(popper_data, debug=True, show_stats=True, quiet=False)
        prog, score, stats = learn_solution(settings)
        prolog = Prolog()
        list(prolog.query(f'unload_file(\'{os.path.abspath(popper_data)}/bk.pl\').'))
        list(prolog.query(f'unload_file(\'{os.path.abspath(popper_data)}/bias.pl\').'))
        list(prolog.query(f'unload_file(\'{os.path.abspath(popper_data)}/exs.pl\').'))
        theory = None if prog is None else format_prog(order_prog(prog))
        if theory is not None:
            stats = eval_rule(theory=theory, ds_train=train_path, ds_val=val_path, dir='TrainGenerator/',
                              print_stats=False, )
            TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = stats
        else:
            TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = [1] * 8
        if print_stats:
            log_stats(stats, path, theory)
        # if prog is not None and print_stats:
        #     print_prog_score(prog, score)
        return theory, TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train

    def aleph_train(self, path, noisy_samples=0, print_stats=True):
        aleph_path = f"{path}/aleph"
        train_path = f'{path}/train_samples.txt'
        val_path = f'{path}/val_samples.txt'
        aleph = Aleph()
        aleph.settingsAsFacts(
            'set(i,2), set(clauselength,10), set(minacc,0.7), set(minscore,3), set(minpos,3),'
            ' set(nodes,5000), set(explore,true), set(max_features, 10)')
        aleph.set('minacc', 0.7)
        # if noisy_samples > 0:
        #     aleph.set('noise', noisy_samples)
        # for s in []
        # aleph.set()
        theory, features = None, None
        with open(aleph_path + '/trains2/train.b') as background, open(
                aleph_path + '/trains2/train.n') as negative, open(aleph_path + '/trains2/train.f') as pos:
            theory, features = aleph.induce('induce', pos.read(), negative.read(),
                                            background.read(), printOutput=False)

        # t = re.split('\s|@', theory)
        # theory = "\n".join([el + '.' for el in t if 'eastbound' in el])
        if theory is not None:
            stats = eval_rule(theory=theory, ds_train=train_path, ds_val=val_path, dir='TrainGenerator/',
                              print_stats=False)
            TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = stats
        else:
            TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = [1] * 8
            raise Warning('Aleph run aborted. No valid theory returned. ')

        if print_stats:
            log_stats(stats, path, theory)
        del aleph
        return theory, TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train

    @staticmethod
    def plot_ilp_crossval(noise):
        visualization.plot_ilp_crossval(noise=noise)

    @staticmethod
    def plot_noise_robustness():
        visualization.plot_noise_robustness()


def log_stats(stats, path, theory):
    TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = stats
    print(f'Aleph, train samples: {path.split("/")[-2].split("_")[-1]},'
          f' decision rule: {path.split("/")[-2].split("_")[-2]}, cv it: {path.split("/")[-1][-1]}')
    print(theory)
    # print(features)
    print(f'training: ACC:{(TP_train + TN_train) / (FN_train + TN_train + TP_train + FP_train)}, '
          f'Precision:{(TP_train / (TP_train + FP_train)) if (TP_train + FP_train) > 0 else 0}, '
          f'Recall:{(TP_train / (TP_train + FN_train)) if (TP_train + FN_train) > 0 else 0}, '
          f'TP:{TP_train}, FN:{FN_train}, TN:{TN_train}, FP:{FP_train}')
    print(f'Validation: ACC:{(TP + TN) / (FN + TN + TP + FP)}, '
          f'Precision:{(TP / (TP + FP)) if (TP + FP) > 0 else 0}, '
          f'Recall:{(TP / (TP + FN)) if (TP + FN) > 0 else 0}, '
          f'TP:{TP}, FN:{FN}, TN:{TN}, FP:{FP}')
    print(f'#################################################')
