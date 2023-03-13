import glob
import os
import warnings
from itertools import product

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_ilp_neural_data
from visualization.vis_util import make_3_im_legend


def image_noise_plot(neural_path, ilp_pth, out_path, training_samples=1000, vis='Trains'):
    labelsize, fontsize = 15, 20
    ilp_stats_path = f'{ilp_pth}/stats'
    neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    data, ilp_models, neural_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path, vis)
    models = neural_models + ilp_models

    data = data.loc[data['training samples'] == training_samples].loc[data['noise type'] != 'label noise']

    scenes = data['scene'].unique()
    rules = data['rule'].unique()
    noise = sorted([str(int(d * 100)) + '%' for d in list(data['noise'].unique())])

    colors_s = sns.color_palette()[:len(noise)]
    colors = {noi: colors_s[n] for n, noi in enumerate(noise)}
    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{out_path}/noise'
    materials_s = ["///", "//", '/', '\\', '\\\\']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    sns.set_theme(style="whitegrid")
    for c, rule in enumerate(rules):
        ax = axes[c // 2, c % 2]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            sns.barplot(x='Methods', y='Validation acc', hue='noise', data=data_temp,
                        palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                        )
        ax.get_legend().remove()
        ax.set_ylim([50, 100])
        ax.get_xaxis().set_visible(False)
        if c % 2:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            # ax.set_yticklabels([''] * 9)
        else:
            ax.set_ylabel('Accuracy', fontsize=fontsize)

    make_3_im_legend(fig, axes, noise, 'Label Noise', models, colors, mt)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/image_noise_{training_samples}_tr_samples.png', bbox_inches='tight', dpi=400)

    plt.close()
