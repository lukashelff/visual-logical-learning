import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_ilp_neural_data
from visualization.vis_util import make_3_im_legend


def generalization_plot(neural_path, ilp_pth, vis='Trains', min_cars=7, max_cars=7, tr_samples=10000):
    labelsize, fontsize = 15, 20

    with open(neural_path + f'/generalization/ilp_generalization_{min_cars}_{max_cars}.csv', 'r') as f:
        data_gen_ilp = pd.read_csv(f)
    data_gen_ilp['Train length'] = '7'
    data_gen_ilp['noise'] = 0
    data_gen_ilp['Validation acc'] = data_gen_ilp['Validation acc'].apply(lambda x: x * 100)

    with open(neural_path + f'/generalization/cnn_generalization_{min_cars}_{max_cars}.csv', 'r') as f:
        data_gen = pd.read_csv(f)
    data_gen = data_gen.rename({'number of images': 'training samples'}, axis='columns')
    data_gen['Train length'] = '7'
    data_gen['noise'] = 0
    data_gen['Validation acc'] = data_gen['Validation acc'].apply(lambda x: x * 100)

    neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    ilp_stats_path = f'{ilp_pth}/stats'
    data, ilp_models, neural_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path, vis)
    data['Train length'] = '2-4'

    data = pd.concat([data, data_gen_ilp, data_gen], ignore_index=True)
    data = data.loc[data['training samples'] == tr_samples].loc[data['noise'] == 0]

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    datasets = list(data['Train length'].unique())
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    models = np.append(neural_models, ilp_models)
    colors_s = sns.color_palette()[:len(datasets)]
    colors = {ds: colors_s[n] for n, ds in enumerate(datasets)}

    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{neural_path}/generalization'
    materials_s = ["///", "//", '/', '\\', '\\\\']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(14, 7))
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

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
            sns.barplot(x='Methods', y='Validation acc', hue='Train length', data=data_temp,
                        palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                        )
        ax.get_legend().remove()
        ax.set_ylim([50, 100])
        ax.get_xaxis().set_visible(False)
        if c % 2:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=fontsize)

    make_3_im_legend(fig, axes, datasets, 'Train length', models, colors, mt)
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/generalization_{tr_samples}_tr_samples.png', bbox_inches='tight', dpi=400)

    plt.close()
