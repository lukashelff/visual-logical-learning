import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_ilp_neural_data


def rule_complexity_plot(neural_path, ilp_pth, vis='Trains', im_count=1000):
    labelsize, fontsize = 15, 20
    ilp_stats_path = f'{ilp_pth}/stats'
    neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    data, ilp_models, neural_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path, vis)
    models = neural_models + ilp_models
    data = data.loc[data['noise'] == 0].loc[data['training samples'] == im_count]

    scenes = data['scene'].unique()
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{neural_path}/rule_complexity'
    materials_s = ["///", "//", '/', '\\', '\\\\']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(12, 7))
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
            sns.barplot(x='Methods', y='Validation acc', hue='training samples', data=data_temp,
                        palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                        )
        ax.get_legend().remove()
        ax.set_ylim([50, 100])
        ax.get_xaxis().set_visible(False)
        if c % 2:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=fontsize)

    axes[1, 1].set_axis_off()
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    plt.rcParams.update({'hatch.color': 'black'})

    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    leg = fig.legend(
        white +
        handels,
        ['Models:'] + [m.title() for m in models],
        loc='lower left',
        bbox_to_anchor=(.515, 0.248),
        frameon=True,
        handletextpad=0,
        ncol=1, handleheight=1.2, handlelength=2.5
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/rules_{im_count}_sample.png', bbox_inches='tight', dpi=400)

    plt.close()


def example(neural_path, ilp_pth, vis='Trains'):
    ilp_stats_path = f'{ilp_pth}/stats'

    dirs = glob.glob(ilp_stats_path + '/*.csv')
    ilp_data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            ilp_data.append(pd.read_csv(f))
    ilp_data = pd.concat(ilp_data, ignore_index=True)
    ilp_models = sorted(ilp_data['Methods'].unique())

    with open(neural_path + '/label_acc_over_epoch.csv', 'r') as f:
        neur_data = pd.read_csv(f)
        neur_data = neur_data.loc[neur_data['epoch'] == 24].loc[neur_data['visualization'] == vis]
    neur_data = neur_data.rename({'number of images': 'training samples'}, axis='columns')
    neural_models = sorted(neur_data['Methods'].unique())

    data = pd.concat([ilp_data, neur_data])
    # data = data.loc[(data['training samples'] == im_count)]
    data['Validation acc'] = data['Validation acc'].apply(lambda x: x * 100)
    data = data.loc[data['noise'] == 0]

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    models = neural_models + ilp_models
    colors_s = sns.color_palette()[:len(im_count)]
    colors = {count: colors_s[n] for n, count in enumerate(im_count)}
    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{neural_path}/rules'
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
        ax.set_title(rule.title(), fontsize=20)
        # ax.titlesize = 25
        ax.tick_params(bottom=False, left=False, labelsize=15)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            sns.barplot(x='Methods', y='Validation acc', hue='training samples', data=data_temp,
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
            ax.set_ylabel('Accuracy', fontsize=15)

    axes[1, 1].set_axis_off()
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    plt.rcParams.update({'hatch.color': 'black'})

    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    leg = fig.legend(
        white + color_markers + white * (len(models) - len(im_count) + 1) + handels,
        ['Training samples:'] + im_count + [''] * (len(models) - len(im_count)) + ['Models:'] + [m.title() for m in
                                                                                                 models],
        loc='lower left',
        bbox_to_anchor=(.515, 0.21),
        frameon=True,
        handletextpad=0,
        ncol=2, handleheight=1.2, handlelength=2.5
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/{vis}_bar_neural_rules.png', bbox_inches='tight', dpi=400)

    plt.close()
