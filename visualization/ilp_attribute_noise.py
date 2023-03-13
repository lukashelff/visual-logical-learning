import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from raw.concept_tester import eval_rule
from visualization.data_handler import get_ilp_neural_data
from visualization.vis_util import make_3_im_legend


def attribute_noise_plot(neural_path, ilp_pth, outpath, training_samples=1000, vis='Trains'):
    labelsize, fontsize = 15, 20

    ilp_stats_path = f'{ilp_pth}/attr_noise/stats'

    data, ilp_models, _ = get_ilp_neural_data(ilp_stats_path, None, vis='Train')
    data = data.loc[data['training samples'] == training_samples]

    noise = sorted([str(int(d * 100)) + '%' for d in list(data['noise'].unique())])
    colors_s = sns.color_palette()[:len(noise)]
    colors = {noi: colors_s[n] for n, noi in enumerate(noise)}

    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{outpath}/noise'
    materials_s = ["///", "//", '/', '\\', '\\\\']
    mt = {model: materials_s[n] for n, model in enumerate(ilp_models)}
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    sns.set_theme(style="whitegrid")
    for c, rule in enumerate(rules):
        baselines = get_baselines(rule)
        ax = axes[c // 2, c % 2]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        if len(data_t) == 0:
            continue
        for model in ilp_models:
            data_temp = data_t.loc[data['Methods'] == model]
            if len(data_temp) > 0:
                sns.barplot(x='Methods', y='Validation acc', hue='noise', data=data_temp,
                            palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=ilp_models
                            )
        ax.get_legend().remove()
        ax.set_ylim([50, 100])
        ax.get_xaxis().set_visible(False)
        if c % 2:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=fontsize)

    make_3_im_legend(fig, axes, noise, 'Concept Noise', ilp_models, colors, mt, legend_h_offset=0.068)
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/attr_noise_{training_samples}_tr_samples.png', bbox_inches='tight', dpi=400)

    plt.close()


def get_baselines(theory):
    baselines = []
    for noise in [0, 0.1, 0.3]:
        ds_path = f'output/ilp/datasets/{theory}/MichalskiTrains10000_{noise}noise/cv_0/train_samples.txt'
        acc = get_baseline(theory, ds_path, noise)
        baselines.append(acc)
        print(f'{theory} {noise} noise: Accuracy = {round(acc * 100, 2)}%')
    return baselines


def get_baseline(theory, ds_path, noise):
    rules = {
        'theoryx': 'eastbound([Car|Cars]):- (short(Car), closed(Car)); (has_load0(Car,triangle), has_load1(Cars,circle)); eastbound(Cars).',
        'complex': 'eastbound(Train):- has_car(Train,Car), load_num(Car,N1), car_num(Car,N2), has_wheel0(Car,N3), N2 < N1, N2 < N3.\n eastbound(Train):- has_car(Train,Car, N1), has_car(Train,Car2), short(Car), long(Car2), car_color(Car, A), car_color(Car2, A), has_wheel0(Car2,N2), N1 < N2.\n eastbound(Train):- has_car(Train,B), has_car(Train,C), has_car(Train,D), car_color(D,X), car_color(C,Y), car_color(B,Z), X\=Y, Y\=Z, Z\=X.',
        'numerical': 'eastbound(Train):- has_car(Train,Car), load_num(Car,N), car_num(Car,N), has_wheel0(Car,N).\n',
    }
    TP, FN, TN, FP, _, _, _, _ = eval_rule(theory=rules[theory], ds_val=ds_path, print_stats=False, noise=noise)
    acc = (TP + TN) / (TP + TN + FP + FN)
    return acc
