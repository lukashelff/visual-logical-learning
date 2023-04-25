import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_ilp_neural_data, read_csv_stats
from visualization.vis_util import make_3_im_legend, make_1_im_legend


def generalization_plot(outpath, vis='Trains', min_cars=7, max_cars=7, tr_samples=10000):
    labelsize, fontsize = 15, 20
    use_materials = False
    fig_path = f'{outpath}/model_comparison/generalization'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    ilp_stats_path = f'{outpath}/ilp/stats'
    neuro_sym_path = f'{outpath}/neuro-symbolic/stats'

    data_gen_ilp = read_csv_stats(fig_path + f'/ilp_generalization_{min_cars}_{max_cars}.csv',
                                  train_length='7', noise=0, symb=True)
    data_gen_cnn = read_csv_stats(fig_path + f'/cnn_generalization_{min_cars}_{max_cars}.csv',
                                  train_length='7', noise=0, symb=False)
    data_gen_neuro_symbolic = read_csv_stats(
        fig_path + f'/neuro_symbolic_generalization_{min_cars}_{max_cars}.csv',
        train_length='7', noise=0, symb=False)

    # neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    # ilp_stats_path = f'{ilp_pth}/stats'
    # neuro_sym_path = f'{neuro_symbolic_path}/stats'

    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_sym_path, vis)
    data['Train length'] = '2-4'

    data = pd.concat([data, data_gen_ilp, data_gen_cnn, data_gen_neuro_symbolic], ignore_index=True)
    # data = pd.concat([data, data_gen_ilp, data_gen_cnn], ignore_index=True)
    data = data.loc[data['training samples'] == tr_samples].loc[data['noise'] == 0].loc[
        data['visualization'] == 'Michalski']

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    train_lengths = list(data['Train length'].unique())
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    models = neural_models + neuro_symbolic_models + ilp_models

    colors_s = sns.color_palette()
    colors_category = train_lengths if use_materials else models
    colors_category_name = 'Train Length' if use_materials else 'Models'
    colors = {vis: colors_s[n] for n, vis in enumerate(colors_category)}

    rules = ['theoryx', 'numerical', 'complex']

    materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '+', 'o', 'O', '.', '*'] if use_materials else ["//", '\\\\',
                                                                                                        'x', '+', "///",
                                                                                                        '/', '\\', 'o',
                                                                                                        'O', '.', '*']
    material_category = models if use_materials else train_lengths
    material_category_name = 'Models' if use_materials else 'Train Length'
    mt = {model: materials_s[n] for n, model in enumerate(material_category)}

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(20, 2))
    gs = fig.add_gridspec(1, 3, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for c, rule in enumerate(rules):
        ax = axes[c]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        if use_materials:
            for model in models:
                data_temp = data_t.loc[data['Methods'] == model]
                try:
                    sns.barplot(x='Methods', y='Validation acc', hue='Train length', data=data_temp, hue_order=train_lengths,
                                palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                                )
                except:
                    pass
        else:
            for ds in train_lengths:
                data_temp = data_t.loc[data['Train length'] == ds]
                try:
                    sns.barplot(x='Train length', y='Validation acc', hue='Methods', data=data_temp, hue_order=models,
                                palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[ds], order=train_lengths
                                )
                except:
                    pass

        for container in ax.containers:
            ax.bar_label(container, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
        ax.get_legend().remove()
        ax.set_ylim([50, 111])
        ax.get_xaxis().set_visible(False)
        if c != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=labelsize)

    # make_3_im_legend(fig, axes, train_lengths, 'Train Length', models, colors, mt)

    offset = -.2

    # make_1_im_legend(fig, axes, train_lengths, 'Train Length', models, colors, mt, legend_h_offset=offset, ncols=6)
    make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
                     labelsize, legend_h_offset=-0.18, legend_v_offset=0.)

    os.makedirs(fig_path, exist_ok=True)
    plt.savefig(fig_path + f'/generalization_{tr_samples}_tr_samples.png', bbox_inches='tight', dpi=400)

    plt.close()
