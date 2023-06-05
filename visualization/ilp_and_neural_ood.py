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

from visualization.data_handler import get_ilp_neural_data, read_csv_stats
from visualization.vis_util import make_3_im_legend, make_1_im_legend, make_1_line_im, make_3_im


def ood_plot(outpath, vis='Trains'):
    use_materials = False
    fig_path = f'{outpath}/model_comparison/ood/ood'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    train_vis = 'Trains'
    train_type = 'RandomTrains'
    base_scene = 'base_scene'

    ood_data = read_csv_stats(outpath + f'/neural/ood/ood_{train_vis}_{train_type}_{base_scene}_len_2-4.csv',
                              train_length='2-4', noise=0, symb=False)
    ood_data['Distribution'] = 'Uniform'
    ood_data['Validation acc'] = ood_data['Validation acc'] / 100

    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(None, neural_stats_path,
                                                                                 None, None, vis)
    data = data.loc[data['image noise'] == 0].loc[data['label noise'] == 0].loc[data['visualization'] == 'Michalski']
    data['Distribution'] = 'Michalski'
    data = pd.concat([data, ood_data], ignore_index=True)
    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    train_lengths = list(data['Train length'].unique())
    distributions = list(data['Distribution'].unique())
    rules = data['rule'].unique()
    models = list(data['Models'].unique())

    colors_category = models
    colors_category_name = 'Models'

    material_category = distributions
    material_category_name = 'Distribution'


    legend_offset = (0.43, 0.213)
    legend_cols = 4
    figsize = (26, 4)
    labelsize, fontsize = 15, 20
    materials_s = ["//", '\\\\', 'x', "///", '/', '\\', '.', 'o', '+', 'O', '*']
    mt = {model: materials_s[n] for n, model in enumerate(material_category)}

    colors_s = sns.color_palette('dark')
    colors = {vis: colors_s[n] for n, vis in enumerate(colors_category)}

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.3)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    completed_runs = []
    data_regimes = {
        100: 'small data regime',
        1000: 'medium data regime',
        10000: 'large data regime',
    }
    for col, tr_samples in enumerate(im_count):
        ax = axes[col // 2, col % 2]
        ax.grid(axis='x')
        ax.set_title(data_regimes[int(tr_samples)].title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['training samples'] == tr_samples]
        run = []
        y_pos = []
        for c, m in product(colors_category, material_category):
            data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
            run += ['*'] if len(data_temp) < 5 else ['']
            y_pos += [40 if np.isnan(data_temp['Validation acc'].mean()) else data_temp['Validation acc'].mean()]
            try:
                sns.barplot(x=colors_category_name, order=colors_category, y='Validation acc',
                            hue=material_category_name, hue_order=material_category, data=data_temp,
                            palette={m: col for m, col in zip(material_category, ['gray' for _ in material_category])},
                            alpha=.7, ax=ax, orient='v', hatch=mt[m])
            except:
                warnings.warn(f'No data for {c}, {m}, {tr_samples}')
        completed_runs += [run]
        for bar_group, desaturate_value in zip(ax.containers, [1] * len(ax.containers)):
            ax.bar_label(bar_group, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=1)
            for c_bar, bar in enumerate(bar_group):
                color = colors_s[c_bar]
                # bar.set_color(sns.desaturate(color, desaturate_value))
                bar.set_facecolor(sns.desaturate(color, desaturate_value))
                bar.set_edgecolor('black')
        ax.get_legend().remove()
        ax.get_xaxis().set_visible(False)
        ax.set_xlabel('')
        x_pos = sorted(list(set([p.get_x() + p.get_width() / 2 for p in ax.patches])))
        if col % 2:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=labelsize)
            ax.set_ylim([50, 111])
        for x, y, r in zip(x_pos, y_pos, run):
            ax.text(x, y + 10, r, fontsize=labelsize, ha='center')
    # axes[1, 1].set_axis_off()
    leg = axes[1, 1]
    leg.get_xaxis().set_visible(False)
    leg.get_yaxis().set_visible(False)

    make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
                     labelsize, legend_h_offset=legend_offset[0], legend_v_offset=legend_offset[1], ncols=legend_cols)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=400)
    # save completed runs
    with open(fig_path.replace('.png', '_completed_runs.txt'), 'w') as f:
        f.write(f'Completed runs: {completed_runs}')
    print(f'run {fig_path} completed')
    print(f'Completed runs: {completed_runs}')

    plt.close()
