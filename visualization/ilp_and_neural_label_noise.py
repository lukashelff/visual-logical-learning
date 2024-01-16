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
from visualization.vis_util import make_3_im_legend, make_1_im_legend, make_1_line_im, make_3_im, make_3_im_deg


def label_noise_plot(outpath, training_samples=1000, vis='Trains'):
    use_materials = False
    fig_path = f'{outpath}/model_comparison/label_noise'

    ilp_stats_path = f'{outpath}/ilp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    neuro_sym_path = f'{outpath}/neuro-symbolic/stats'
    alpha_ilp = f'{outpath}/neuro-symbolic/alphailp/stats'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_sym_path, alpha_ilp, vis)
    models = neural_models + neuro_symbolic_models + ilp_models

    data = data.loc[data['training samples'] == training_samples].loc[data['image noise'] == 0].loc[
        data['visualization'] == 'Michalski'].loc[data['Train length'] == '2-4']

    scenes = data['scene'].unique()
    rules = data['rule'].unique()
    data['label noise'] = data['label noise'].apply(lambda x: str(int(x * 100)) + '%')
    noise = sorted(data['label noise'].unique())

    colors_category = noise if use_materials else models
    colors_category_name = 'label noise' if use_materials else 'Models'

    material_category = models if use_materials else noise
    material_category_name = 'Models' if use_materials else 'label noise'

    # make_1_line_im(data, material_category, material_category_name, colors_category, colors_category_name,
    #                fig_path + f'/label_noise_{training_samples}_tr_samples.png', (27, 2))
    pth = fig_path + f'/label_noise.png' if training_samples == 1000 else fig_path + f'/label_noise_{training_samples}_tr_samples.png'
    make_3_im(data, material_category, material_category_name, colors_category, colors_category_name,
              pth, (30, 8), legend_offset=(0.05, 0), legend_cols=4)
    # (27, 4), legend_offset=(0.43, 0.213), legend_cols=4)


def label_noise_degradation_plot(outpath, training_samples=1000, vis='Trains'):
    use_materials = False
    ilp_stats_path = f'{outpath}/ilp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    alpha_ilp = f'{outpath}/neuro-symbolic/alphailp/stats'
    neuro_sym_path = f'{outpath}/neuro-symbolic/stats'
    fig_path = f'{outpath}/model_comparison/label_noise_degradation'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_sym_path, alpha_ilp, vis)
    models = neural_models + neuro_symbolic_models + ilp_models
    data = data.loc[data['training samples'] == training_samples].loc[data['image noise'] == 0].loc[
        data['visualization'] == 'Michalski'].loc[data['Train length'] == '2-4']

    rules = data['rule'].unique()
    rules = ['theoryx', 'numerical', 'complex']
    scenes = data['scene'].unique()
    iterations = data['cv iteration'].unique()
    # initialize column with nan
    data['Noise Degradation'] = np.nan

    for rule, model, n, cv in product(rules, models, [0.1, 0.3], iterations):
        n_idx = data.loc[data['rule'] == rule].loc[data['label noise'] == n].loc[data['Models'] == model].loc[
            data['cv iteration'] == cv].index
        idx = data.loc[data['rule'] == rule].loc[data['label noise'] == 0].loc[data['Models'] == model].loc[
            data['cv iteration'] == cv]['Validation acc'].index

        if len(idx) > 0 and len(n_idx) > 0:
            n_acc = data.loc[n_idx, 'Validation acc'] - 50
            acc = data.loc[idx, 'Validation acc'] - 50
            # print(f'acc: {acc.values}, noise acc: {n_acc.values}, set: {rule}, {model}, {n}, {cv}')
            data.loc[n_idx, 'Noise Degradation'] = (acc.values - n_acc.values) / acc.values * 100
        else:
            warnings.warn(
                f'No data for {rule}, {model}, {n} noise, iteration {cv}, {training_samples} training samples')
    data = data.loc[data['label noise'] != 0]
    data['label noise'] = data['label noise'].apply(lambda x: str(int(x * 100)) + '%')
    noise = sorted(data['label noise'].unique())

    colors_category = noise if use_materials else models
    colors_category_name = 'label noise' if use_materials else 'Models'
    material_category = models if use_materials else noise
    material_category_name = 'Models' if use_materials else 'label noise'
    pth = fig_path + f'/label_noise_degradation.png' if training_samples == 1000 else fig_path + f'/label_noise_degradation_{training_samples}_tr_samples.png'
    make_3_im_deg(data, material_category, material_category_name, colors_category, colors_category_name, pth, (27, 4),
                  legend_offset=(0.43, 0.213), legend_cols=4)

    # materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '+', 'o', 'O', '.', '*'] if use_materials else ["//", '\\\\',
    #                                                                                                     'x', '+', "///",
    #                                                                                                     '/', '\\', 'o',
    #                                                                                                     'O', '.', '*']
    # mt = {model: materials_s[n] for n, model in enumerate(material_category)}
    #
    # sns.set_theme(style="whitegrid")
    # fig = plt.figure(figsize=(27, 2))
    # gs = fig.add_gridspec(1, 3, wspace=.05, hspace=.15)
    # axes = gs.subplots(sharex=True, sharey=True, )
    # axes = axes if isinstance(axes, np.ndarray) else [axes]
    #
    # for col, rule in enumerate(rules):
    #     ax = axes[col]
    #     ax.grid(axis='x')
    #     ax.set_title(rule.title(), fontsize=fontsize)
    #     ax.tick_params(bottom=False, left=False, labelsize=labelsize)
    #     for spine in ax.spines.values():
    #         spine.set_edgecolor('gray')
    #     data_t = data.loc[data['rule'] == rule]
    #     if data_t.empty:
    #         continue
    #     # for model in models:
    #     #     data_temp = data_t.loc[data['Methods'] == model]
    #     #     if len(data_temp) > 0:
    #     #         sns.barplot(x='Methods', y='Noise Degradation', hue='noise', data=data_temp,
    #     #                     palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
    #     #                     )
    #     for c, m in product(colors_category, material_category):
    #         data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
    #         try:
    #             sns.barplot(x=colors_category_name, order=colors_category, y='Noise Degradation',
    #                         hue=material_category_name, hue_order=material_category, data=data_temp,
    #                         palette={m: col for m, col in zip(material_category, ['gray' for _ in material_category])},
    #                         alpha=.7, ax=ax, orient='v', hatch=mt[m])
    #         except:
    #             warnings.warn(f'No data for {c}, {m}, {rule}')
    #     # for c, m in product(colors_category, material_category):
    #     #     data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
    #     #     try:
    #     #         sns.barplot(x=material_category_name, order=material_category, y='Noise Degradation',
    #     #                     hue=colors_category_name,
    #     #                     hue_order=colors_category, data=data_temp, palette="dark", alpha=.7, ax=ax, orient='v',
    #     #                     hatch=mt[m])
    #     #     except:
    #     #         warnings.warn(f'No data for {c}, {m}, {rule}')
    #
    #     for bar_group, desaturate_value in zip(ax.containers, [1] * len(ax.containers)):
    #         ax.bar_label(bar_group, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
    #         for c_bar, bar in enumerate(bar_group):
    #             color = colors_s[c_bar]
    #             # bar.set_color(sns.desaturate(color, desaturate_value))
    #             bar.set_facecolor(sns.desaturate(color, desaturate_value))
    #             bar.set_edgecolor('black')
    #     ax.get_legend().remove()
    #     ax.set_ylim([0, 111])
    #     ax.get_xaxis().set_visible(False)
    #     if col != 0:
    #         ax.set_ylabel('')
    #     else:
    #         ax.set_ylabel('Degradation (%)', fontsize=labelsize)
    #
    # make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
    #                  labelsize, legend_h_offset=-0.18, legend_v_offset=0.)
    # os.makedirs(fig_path, exist_ok=True)
    # plt.savefig(fig_path + f'/label_noise_{training_samples}_tr_samples_acc_loss.png', bbox_inches='tight', dpi=400)
    #
    # plt.close()
