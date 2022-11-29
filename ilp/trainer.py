import os
import random
import re

import pyswip
import seaborn as sns

import numpy as np

import pandas as pd
from matplotlib import pyplot as plt
from popper.loop import learn_solution
from popper.util import Settings, format_prog, order_prog, print_prog_score

import popper

import matplotlib.lines as mlines
from tabulate import tabulate

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
                  train_count=[100, 1000, 10000], ds_size=12000):
        data = pd.DataFrame(
            columns=['Methods', 'training samples', 'rule', 'cv iteration', 'Validation acc', 'theory'])
        ilp_stats_path = f'output/ilp/stats/'
        os.makedirs(ilp_stats_path, exist_ok=True)
        for model in models:
            for class_rule in rules:
                ds_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{raw_trains}_{class_rule}.txt'
                out_path = f'output/ilp/datasets/{raw_trains}_{class_rule}'
                os.makedirs(out_path, exist_ok=True)
                popper_path = f'{out_path}/popper/gt1'
                aleph_path = f"{out_path}/aleph"
                train_path = f'{out_path}/train_samples.txt'
                val_path = f'{out_path}/val_samples.txt'
                # ds_size = 10
                noise = 0.0
                for train_size in train_count:
                    with open(ds_path, "r") as file:
                        all_data = file.readlines()
                        if len(all_data) != ds_size:
                            raise f'datasets of size {ds_size} however only {len(all_data)} datasamples were generated'
                        y = [l[0] for l in all_data]
                        sss = StratifiedShuffleSplit(n_splits=folds, train_size=train_size, test_size=2000,
                                                     random_state=0)
                        li = []
                        for fold, (tr_idx, val_idx) in enumerate(sss.split(np.zeros(len(y)), y)):
                            train_samples = map(all_data.__getitem__, tr_idx)
                            val_samples = map(all_data.__getitem__, val_idx)
                            with open(train_path, 'w+') as train, open(val_path, 'w+') as val:
                                train.writelines(train_samples)
                                val.writelines(val_samples)
                            create_bk(train_path, out_path, train_size, noise)

                            if model == 'popper':
                                theory = self.popper_train(popper_path, print_stats=True)
                            elif model == 'aleph':
                                theory = self.aleph_train(aleph_path)
                            else:
                                raise ValueError(f'model: {model} not supported')

                            # print(theory)
                            if theory is not None:
                                TP_train, FN_train, TN_train, FP_train = eval_rule(theory=theory, ds=train_path,
                                                                                   dir='TrainGenerator/',
                                                                                   print_stats=False)
                                TP, FN, TN, FP = eval_rule(theory=theory, ds=val_path, dir='TrainGenerator/',
                                                           print_stats=True)
                            else:
                                TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = [1] * 8
                            li.append([model, train_size, class_rule, fold,
                                       (TP + TN) / (TP + FN + TN + FP),
                                       (TP_train + TN_train) / (TP_train + FN_train + TN_train + FP_train), theory])
                        _df = pd.DataFrame(li, columns=['Methods', 'training samples', 'rule', 'cv iteration',
                                                        'Validation acc', 'Train acc', 'theory'])
                        data = pd.concat([data, _df], ignore_index=True)

        data.to_csv(ilp_stats_path + '/ilp_stats.csv')

    def train(self, model, raw_trains, class_rule, train_size, val_size):
        ds_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{raw_trains}_{class_rule}.txt'
        out_path = f'output/ilp/datasets/{raw_trains}_{class_rule}'
        os.makedirs(out_path, exist_ok=True)
        popper_path = f'{out_path}/popper/gt1'
        aleph_path = f"{out_path}/aleph"
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
                theory = self.popper_train(popper_path, print_stats=True)
            elif model == 'aleph':
                theory = self.aleph_train(aleph_path, print_stats=True)
            else:
                raise ValueError(f'model: {model} not supported')

            TP_train, FN_train, TN_train, FP_train = eval_rule(theory=theory, ds=train_path,
                                                               dir='TrainGenerator/', clean_up=False)
            TP_train, FN_train, TN_train, FP_train = eval_rule(theory=theory, ds=val_path,
                                                               dir='TrainGenerator/', clean_up=False)

    def popper_train(self, popper_data, print_stats=False):
        import popper
        popper = reload(popper)
        # reload(popper.loop.learn_solution)
        prog, score, stats = popper.loop.learn_solution(
            Settings(popper_data, debug=False, show_stats=False, quiet=True))
        theory = None if prog is None else format_prog(order_prog(prog))
        if prog is not None and print_stats:
             print_prog_score(prog, score)
        return theory

    def aleph_train(self, aleph_path, print_stats=False):
        aleph = Aleph()
        aleph.settingsAsFacts(
            'set(i,2), set(clauselength,10), set(minacc,0.6), set(minscore,3), set(minpos,3),'
            ' set(nodes,5000), set(explore,true), set(max_features,10)')
        with open(aleph_path + '/trains2/train.b') as background, open(
                aleph_path + '/trains2/train.n') as negative, open(aleph_path + '/trains2/train.f') as pos:
            theory, features = aleph.induce('induce_features', pos.read(), negative.read(),
                                            background.read())

        t = re.split('\s|@', theory)
        theory = "\n".join([el + '.' for el in t if 'eastbound' in el])
        if print_stats:
            print(theory)
            print(features)
        return theory

    def plot_ilp_crossval(self):
        ilp_stats_path = f'output/ilp/stats'
        ilp_vis_path = f'output/ilp/vis'
        with open(ilp_stats_path + '/ilp_stats.csv', 'r') as f:
            data = pd.read_csv(f)

        model_names = data['Methods'].unique()

        fig = plt.figure()
        gs = fig.add_gridspec(len(model_names), hspace=0)
        axes = gs.subplots(sharex=True, sharey=True)
        axes = axes if isinstance(axes, np.ndarray) else [axes]
        im_count = sorted(data['training samples'].unique())
        colors_s = sns.color_palette()[:len(im_count) + 1]
        colors = {count: colors_s[n] for n, count in enumerate(im_count)}
        # colors[count] = colors_s[n]
        # colors = {
        #     10000: colors_s[2],
        #     1000: colors_s[1],
        #     100: colors_s[0]
        # }
        rules = data['rule'].unique()
        for rule, ax in zip(rules, axes):
            ax.grid(axis='x', linestyle='solid', color='gray')
            ax.tick_params(bottom=False, left=False)
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
            data_t = data.loc[data['rule'] == rule]
            # sns.violinplot(x='Validation acc', y='rule', hue='number of images', data=data_t,
            #                inner="quart", linewidth=0.5, dodge=False, palette="pastel", saturation=.2, scale='width',
            #                ax=ax
            #                )
            for count in im_count:
                data_tmp = data_t.loc[data_t['training samples'] == count]
                # for count in im_count:
                #     data_tmp = data_t.loc[data_t['number of images'] == count]
                #     print(tabulate(data_tmp == data.loc[data['epoch'] == 24].loc[data['Methods'] == 'resnet18'].loc[data['visualization'] == vis].loc[data['number of images'] == count].loc[data['epoch'] == 24], headers='keys', tablefmt='psql'))
                # Show each observation with a scatterplot
                tabulate(data_tmp)
                sns.stripplot(x='Validation acc', y='Methods',
                              data=data_tmp,
                              dodge=True,
                              alpha=.25,
                              zorder=1,
                              jitter=False,
                              palette=[colors[count], colors[count]],
                              ax=ax
                              )

            # Show the conditional means, aligning each pointplot in the
            # center of the strips by adjusting the width allotted to each
            # category (.8 by default) by the number of hue levels

            sns.pointplot(x='Validation acc', y='Methods', hue='training samples', data=data_t,
                          dodge=False,
                          join=False,
                          # palette="dark",
                          markers="d",
                          scale=.75,
                          errorbar=None,
                          ax=ax
                          )
        # plt.title('Comparison of Supervised learning methods')
        # Improve the legend
        handles, labels = axes[-1].get_legend_handles_labels()
        length = len(handles) // 2
        # length = 0
        # simple = mlines.Line2D([], [], color='grey', marker='X', linestyle='None', markersize=5)
        # simple_lab = 'Simple'
        trains = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5)
        trains_lab = 'Train acc'
        mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
        mean_lab = 'Mean'
        color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                         im_count]
        for c, ax in enumerate(axes):
            ax.get_legend().remove()
            ax.set_xlim([0.5, 1])
            ax.set_ylabel(model_names[c])
            ax.set_ylabel(rules[c])

        axes[-1].legend([trains, mean] + color_markers,
                        [trains_lab, mean_lab] + [str(i) for i in im_count], title="Training samples",
                        loc='lower center', bbox_to_anchor=(1.2, 0), frameon=False,
                        handletextpad=0, ncol=2)

        # axes[-1].set_ylabel(f"Rule-based learning problem")
        axes[-1].set_xlabel("Validation accuracy")

        os.makedirs(ilp_vis_path, exist_ok=True)
        plt.savefig(ilp_vis_path + f'/ilp_mean_variance.png', bbox_inches='tight', dpi=400)

        plt.close()
