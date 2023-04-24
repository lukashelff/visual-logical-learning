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


def generalization_plot(neural_path, ilp_pth,neuro_symbolic_path, outpath, vis='Trains', min_cars=7, max_cars=7, tr_samples=10000):
    labelsize, fontsize = 15, 20
    data_gen_ilp = read_csv_stats(outpath + f'/generalization/ilp_generalization_{min_cars}_{max_cars}.csv',
                                  train_length='7', noise=0, symb=True)
    data_gen_cnn = read_csv_stats(outpath + f'/generalization/cnn_generalization_{min_cars}_{max_cars}.csv',
                                  train_length='7', noise=0, symb=False)
    data_gen_neuro_symbolic = read_csv_stats(
        outpath + f'/generalization/neuro_symbolic_generalization_{min_cars}_{max_cars}.csv',
        train_length='7', noise=0, symb=False)

    neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    ilp_stats_path = f'{ilp_pth}/stats'
    neuro_sym_path = f'{neuro_symbolic_path}/stats'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path, None, vis)
    data['Train length'] = '2-4'

    # data = pd.concat([data, data_gen_ilp, data_gen_cnn, data_gen_neuro_symbolic], ignore_index=True)
    data = pd.concat([data, data_gen_ilp, data_gen_cnn], ignore_index=True)
    data = data.loc[data['training samples'] == tr_samples].loc[data['noise'] == 0].loc[data['visualization'] == 'Michalski']

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    datasets = list(data['Train length'].unique())
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    models = np.append(neural_models, neuro_symbolic_models + ilp_models)
    colors_s = sns.color_palette()[:len(datasets)]
    colors = {ds: colors_s[n] for n, ds in enumerate(datasets)}

    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{outpath}/generalization'
    materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '+', 'o', 'O', '.', '*']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
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
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            try:
                sns.barplot(x='Methods', y='Validation acc', hue='Train length', data=data_temp, hue_order=datasets,
                            palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
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

    # make_3_im_legend(fig, axes, datasets, 'Train Length', models, colors, mt)

    make_1_im_legend(fig, axes, datasets, 'Train Length', models, colors, mt, legend_h_offset=0, ncols=6)
    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/generalization_{tr_samples}_tr_samples.png', bbox_inches='tight', dpi=400)

    plt.close()
