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


def label_noise_plot(outpath, training_samples=1000, vis='Trains'):
    labelsize, fontsize = 15, 20
    use_materials = False
    ilp_stats_path = f'{outpath}/ilp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    fig_path = f'{outpath}/model_comparison/label_noise'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 None, vis)
    models = neural_models + ilp_models

    data = data.loc[data['training samples'] == training_samples]

    scenes = data['scene'].unique()
    rules = data['rule'].unique()
    data['noise'] = data['noise'].apply(lambda x: str(int(x * 100)) + '%')
    noise = sorted(data['noise'].unique())

    colors_s = sns.color_palette()
    colors_category = noise if use_materials else models
    colors_category_name = 'noise' if use_materials else 'Models'
    colors = {noi: colors_s[n] for n, noi in enumerate(colors_category)}
    rules = ['theoryx', 'numerical', 'complex']

    materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '+', 'o', 'O', '.', '*'] if use_materials else ["//", '\\\\',
                                                                                                        'x', '+', "///",
                                                                                                        '/', '\\', 'o',
                                                                                                        'O', '.', '*']
    material_category = models if use_materials else noise
    material_category_name = 'Models' if use_materials else 'noise'
    mt = {model: materials_s[n] for n, model in enumerate(material_category)}

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
        for c, m in product(colors_category, material_category):
            data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
            try:
                sns.barplot(x=material_category_name, order=material_category, y='Validation acc',
                            hue=colors_category_name,
                            hue_order=colors_category, data=data_temp, palette="dark", alpha=.7, ax=ax, orient='v',
                            hatch=mt[m])
            except:
                warnings.warn(f'No data for {c}, {m}, {rule}')

        for container in ax.containers:
            ax.bar_label(container, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
        try:
            ax.get_legend().remove()
        except:
            pass
        ax.set_ylim([50, 111])
        ax.get_xaxis().set_visible(False)
        if c != 0:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            # ax.set_yticklabels([''] * 9)
        else:
            ax.set_ylabel('Accuracy', fontsize=labelsize)

    make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
                     labelsize, ncols=6, legend_h_offset=-0, legend_v_offset=0.)
    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + f'/label_noise_{training_samples}_tr_samples.png', bbox_inches='tight', dpi=400)

    plt.close()


def label_noise_degradation_plot(outpath, training_samples=1000, vis='Trains'):
    labelsize, fontsize = 15, 20
    use_materials = True
    ilp_stats_path = f'{outpath}/ilp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'  #
    fig_path = f'{outpath}/model_comparison/label_noise_degradation'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 None, vis)
    models = neural_models + ilp_models
    data = data.loc[data['training samples'] == training_samples]

    rules = data['rule'].unique()
    rules = ['theoryx', 'numerical', 'complex']
    scenes = data['scene'].unique()
    iterations = data['cv iteration'].unique()
    data['Noise Degradation'] = 0

    for rule, model, n, cv in product(rules, models, [0.1, 0.3], iterations):
        n_idx = data.loc[data['rule'] == rule].loc[data['noise'] == n].loc[data['Models'] == model].loc[
            data['cv iteration'] == cv].index
        idx = data.loc[data['rule'] == rule].loc[data['noise'] == 0].loc[data['Models'] == model].loc[
            data['cv iteration'] == cv]['Validation acc'].index

        if len(idx) > 0 and len(n_idx) > 0:
            n_acc = data.loc[n_idx, 'Validation acc']
            acc = data.loc[idx, 'Validation acc']
            # print(f'acc: {acc.values}, noise acc: {n_acc.values}, set: {rule}, {model}, {n}, {cv}')
            data.loc[n_idx, 'Noise Degradation'] = (acc.values - n_acc.values) / acc.values * 100
        else:
            warnings.warn(
                f'No data for {rule}, {model}, {n} noise, iteration {cv}, {training_samples} training samples')
    im_count = sorted(data['training samples'].unique())
    data = data.loc[data['noise'] != 0]
    data['noise'] = data['noise'].apply(lambda x: str(int(x * 100)) + '%')
    noise = sorted(data['noise'].unique())

    # colors_s = sns.color_palette()[:len(noise)]
    # colors = {noise[0]: colors_s[0], noise[1]: colors_s[1]}
    # materials_s = ["///", "//", '/', '\\', '\\\\']
    # mt = {model: materials_s[n] for n, model in enumerate(models)}

    colors_s = sns.color_palette()
    colors_category = noise if use_materials else models
    colors_category_name = 'noise' if use_materials else 'Models'
    colors = {noi: colors_s[n] for n, noi in enumerate(colors_category)}
    rules = ['theoryx', 'numerical', 'complex']

    materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '+', 'o', 'O', '.', '*'] if use_materials else ["//", '\\\\',
                                                                                                        'x', '+', "///",
                                                                                                        '/', '\\', 'o',
                                                                                                        'O', '.', '*']
    material_category = models if use_materials else noise
    material_category_name = 'Models' if use_materials else 'noise'
    mt = {model: materials_s[n] for n, model in enumerate(material_category)}

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(27, 2))
    gs = fig.add_gridspec(1, 3, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    sns.set_theme(style="whitegrid")
    for col, rule in enumerate(rules):
        ax = axes[col]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        if data_t.empty:
            continue
        # for model in models:
        #     data_temp = data_t.loc[data['Methods'] == model]
        #     if len(data_temp) > 0:
        #         sns.barplot(x='Methods', y='Noise Degradation', hue='noise', data=data_temp,
        #                     palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
        #                     )

        for c, m in product(colors_category, material_category):
            data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
            try:
                sns.barplot(x=material_category_name, order=material_category, y='Noise Degradation',
                            hue=colors_category_name,
                            hue_order=colors_category, data=data_temp, palette="dark", alpha=.7, ax=ax, orient='v',
                            hatch=mt[m])
            except:
                warnings.warn(f'No data for {c}, {m}, {rule}')

        for container in ax.containers:
            ax.bar_label(container, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
        ax.get_legend().remove()
        ax.set_ylim([0, 60])
        ax.get_xaxis().set_visible(False)
        if col != 0:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            # ax.set_yticklabels([''] * 9)
        else:
            ax.set_ylabel('Loss in Accuracy', fontsize=labelsize)

    make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
                     labelsize, ncols=6, legend_h_offset=-0, legend_v_offset=0.)

    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + f'/label_noise_{training_samples}_tr_samples_acc_loss.png', bbox_inches='tight', dpi=400)

    plt.close()
