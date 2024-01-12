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
from visualization.vis_util import make_3_im_legend, make_1_im_legend, make_1_line_im


def elementary_vs_realistic_plot(outpath, rule='theoryx', tr_samples=1000):
    labelsize = 15
    use_materials = False
    ilp_stats_path = f'{outpath}/ilp/stats'
    neuro_symbolic_stats_path = f'{outpath}/neuro-symbolic/stats'
    alpha_ilp = f'{outpath}/neuro-symbolic/alphailp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    out_dir = f'{outpath}/model_comparison/elementary_vs_realistic'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_symbolic_stats_path, alpha_ilp,
                                                                                 'all')
    models = neural_models + neuro_symbolic_models + ilp_models
    data = data.loc[data['training samples'] == tr_samples].loc[data['label noise'] == 0].loc[
        data['image noise'] == 0].loc[data['rule'] == rule].loc[data['Train length'] == '2-4']

    # duplicate data for symbolic as both visualizations are the same
    data_symbolic = data.loc[data['Models'].isin(ilp_models)].copy()
    data_symbolic['visualization'] = 'Block'
    data = pd.concat([data, data_symbolic], ignore_index=True)

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    rules = data['rule'].unique()

    visualizations = list(data['visualization'].unique())
    # models = np.append(neural_models, ilp_models)
    colors_s = sns.color_palette()
    colors_category = visualizations if use_materials else models
    colors_category_name = 'visualization' if use_materials else 'Models'
    colors = {vis: colors_s[n] for n, vis in enumerate(colors_category)}
    rules = ['theoryx', 'numerical', 'complex']

    materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '.', 'o', '+', 'O', '*'] if use_materials else \
        ["//", '\\\\', 'x', '+', "///", '/', '\\', 'o', 'O', '.', '*']
    material_category = models if use_materials else visualizations
    material_category_name = 'Models' if use_materials else 'visualization'

    make_1_line_im(data, material_category, material_category_name, colors_category, colors_category_name,
                   out_dir + f'/elementary_vs_realistic_{tr_samples}_samples_{rule}.png',
                   figsize=(26, 4), rules=[rule], ncol=4)
    # mt = {model: materials_s[n] for n, model in enumerate(material_category)}
    # sns.set_theme(style="whitegrid")
    # fig = plt.figure(figsize=(16, 2))
    # gs = fig.add_gridspec(1, 1, wspace=.05, hspace=.15)
    # ax = gs.subplots(sharex=True, sharey=True, )
    #
    # sns.set_theme(style="whitegrid")
    # ax.grid(axis='x')
    # # ax.set_title(rule.title(), fontsize=20)
    # ax.tick_params(bottom=False, left=False, labelsize=labelsize)
    # for spine in ax.spines.values():
    #     spine.set_edgecolor('gray')
    # data_t = data.loc[data['rule'] == rule]
    # for c, m in product(colors_category, material_category):
    #     data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
    #     try:
    #         sns.barplot(x=material_category_name, order=material_category, y='Validation acc',
    #                     hue=colors_category_name,
    #                     hue_order=colors_category, data=data_temp, palette="dark", alpha=.7, ax=ax, orient='v',
    #                     hatch=mt[m])
    #     except:
    #         warnings.warn(f'No data for {c}, {m}, {rule}')
    # for container in ax.containers:
    #     ax.bar_label(container, fmt='%.1f', label_type='edge', fontsize=labelsize, padding=3)
    # # if use_materials:
    # ax.get_legend().remove()
    # ax.set_ylim([50, 111])
    # ax.get_xaxis().set_visible(False)
    # ax.set_ylabel('Accuracy', fontsize=labelsize)
    #
    # make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
    #                  labelsize, legend_h_offset=-0.18, legend_v_offset=0.026)
    #
    # os.makedirs(out_dir, exist_ok=True)
    # plt.savefig(out_dir + f'/elementary_vs_realistic_{tr_samples}_samples_{rule}.png', bbox_inches='tight', dpi=400)
    #
    # plt.close()


def elementary_vs_realistic_plot_multi_rule(neural_path, ilp_pth, neuro_symbolic_pth, outpath, tr_samples=1000):
    labelsize, fontsize = 15, 20
    ilp_stats_path = f'{ilp_pth}/stats'
    neuro_symbolic_stats_path = f'{neuro_symbolic_pth}/stats'
    neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_symbolic_stats_path, 'all')
    models = np.append(neural_models, neuro_symbolic_models + ilp_models)
    data = data.loc[data['training samples'] == tr_samples].loc[data['noise'] == 0]

    # duplicate data for symbolic as both visualizations are the same
    data_symbolic = data.loc[data['Methods'].isin(ilp_models)]
    data_symbolic['visualization'] = 'Block'
    data = pd.concat([data, data_symbolic], ignore_index=True)

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    visualizations = list(data['visualization'].unique())
    # models = np.append(neural_models, ilp_models)
    colors_s = sns.color_palette()[:len(visualizations) + 1]
    colors = {vis: colors_s[n] for n, vis in enumerate(visualizations)}
    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{outpath}/elementary_vs_realistic'
    materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '+', 'o', 'O', '.', '*']
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
            sns.barplot(x='Methods', y='Validation acc', hue='visualization', hue_order=visualizations,
                        data=data_temp, palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
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

    make_3_im_legend(fig, axes, visualizations, 'Visualizations', models, colors, mt, legend_h_offset=-0.1)

    os.makedirs(out_path, exist_ok=True)
    pth = out_path + f'/elementary_vs_realistic.png' if tr_samples == 1000 else out_path + f'/elementary_vs_realistic_{tr_samples}_samples.png'
    plt.savefig(pth, bbox_inches='tight', dpi=400)
    print(f'elementary_vs_realistic_{tr_samples}_samples.png saved to {out_path}.')

    plt.close()
