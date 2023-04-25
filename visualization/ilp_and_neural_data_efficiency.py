import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_ilp_neural_data
from visualization.vis_util import make_3_im_legend, make_1_im_legend


def data_efficiency_plot(outpath, vis='Trains'):
    labelsize, fontsize = 15, 20
    # ilp_stats_path = f'{ilp_pth}/stats'
    # neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    ilp_stats_path = f'{outpath}/ilp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    neuro_sym_path = f'{outpath}/neuro-symbolic/stats'
    fig_path = f'{outpath}/model_comparison/data_efficiency'

    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_sym_path, vis)
    models = np.append(neural_models, ilp_models)
    data = data.loc[data['noise'] == 0]

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    im_count = [int(i) for i in im_count]
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    colors_s = sns.color_palette()[:len(im_count)]
    colors = {count: colors_s[n] for n, count in enumerate(im_count)}
    rules = ['theoryx', 'numerical', 'complex']

    materials_s = ["///", "//", '/', '\\', '\\\\']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(27, 2))
    gs = fig.add_gridspec(1, 3, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    sns.set_theme(style="whitegrid")
    for c, rule in enumerate(rules):
        ax = axes[c]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        if data_t.empty:
            continue
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            sns.barplot(x='Methods', y='Validation acc', hue='training samples', data=data_temp,
                        palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                        )
        for container in ax.containers:
            ax.bar_label(container, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
        ax.get_legend().remove()
        ax.set_ylim([50, 111])
        ax.get_xaxis().set_visible(False)
        if c != 0:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            # ax.set_yticklabels([''] * 9)
        else:
            ax.set_ylabel('Accuracy', fontsize=labelsize)

    make_1_im_legend(fig, axes, im_count, 'Training Samples', models, colors, mt, legend_h_offset=0, ncols=6)

    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + f'/data_efficiency.png', bbox_inches='tight', dpi=400)

    plt.close()
