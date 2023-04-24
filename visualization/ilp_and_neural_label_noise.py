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
from visualization.vis_util import make_3_im_legend, make_1_im_legend


def label_noise_plot(neural_path, ilp_pth, outpath, training_samples=1000, vis='Trains'):
    labelsize, fontsize = 15, 20
    ilp_stats_path = f'{ilp_pth}/stats'
    neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path, None, vis)
    models = np.append(neural_models, ilp_models)

    data = data.loc[data['training samples'] == training_samples]

    scenes = data['scene'].unique()
    rules = data['rule'].unique()
    noise = sorted([str(int(d * 100)) + '%' for d in list(data['noise'].unique())])

    colors_s = sns.color_palette()[:len(noise)]
    colors = {noi: colors_s[n] for n, noi in enumerate(noise)}
    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{outpath}/noise'
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
            sns.barplot(x='Methods', y='Validation acc', hue='noise', data=data_temp,
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

    make_1_im_legend(fig, axes, noise, 'Label Noise', models, colors, mt, ncols=6)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/label_noise_{training_samples}_tr_samples.png', bbox_inches='tight', dpi=400)

    plt.close()


def label_noise_degradation_plot(neural_path, ilp_pth,outpath, training_samples=1000, vis='Trains'):
    labelsize, fontsize = 15, 20
    ilp_stats_path = f'{ilp_pth}/stats'
    neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path, None, vis)
    models = np.append(neural_models, ilp_models)
    data = data.loc[data['training samples'] == training_samples]

    rules = data['rule'].unique()
    rules = ['theoryx', 'numerical', 'complex']
    scenes = data['scene'].unique()
    iterations = data['cv iteration'].unique()
    data['Noise Degradation'] = 0

    for rule, model, n, cv in product(rules, models, [0.1, 0.3], iterations):
        n_idx = data.loc[data['rule'] == rule].loc[data['noise'] == n].loc[data['Methods'] == model].loc[
            data['cv iteration'] == cv].index
        idx = data.loc[data['rule'] == rule].loc[data['noise'] == 0].loc[data['Methods'] == model].loc[
            data['cv iteration'] == cv]['Validation acc'].index

        if len(idx) > 0 and len(n_idx) > 0:
            n_acc = data.loc[n_idx, 'Validation acc']
            acc = data.loc[idx, 'Validation acc']
            # print(f'acc: {acc.values}, noise acc: {n_acc.values}, set: {rule}, {model}, {n}, {cv}')
            data.loc[n_idx, 'Noise Degradation'] = acc.values - n_acc.values
        else:
            warnings.warn(
                f'No data for {rule}, {model}, {n} noise, iteration {cv}, {training_samples} training samples')
    noise = sorted([str(int(d * 100)) + '%' for d in list(data['noise'].unique())])[1:]
    im_count = sorted(data['training samples'].unique())
    data = data.loc[data['noise'] != 0]

    colors_s = sns.color_palette()[:len(noise)]
    colors = {noise[0]: colors_s[0], noise[1]: colors_s[1]}
    out_path = f'{outpath}/noise'
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
            if len(data_temp) > 0:
                sns.barplot(x='Methods', y='Noise Degradation', hue='noise', data=data_temp,
                            palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                            )
        for container in ax.containers:
            ax.bar_label(container, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
        ax.get_legend().remove()
        # ax.set_ylim([0.5, 1])
        ax.get_xaxis().set_visible(False)
        if c != 0:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            # ax.set_yticklabels([''] * 9)
        else:
            ax.set_ylabel('Loss in Accuracy', fontsize=fontsize)

    make_1_im_legend(fig, axes, noise, 'Label Noise', models, colors, mt, ncols=6)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/label_noise_{training_samples}_tr_samples_acc_loss.png', bbox_inches='tight', dpi=400)

    plt.close()
