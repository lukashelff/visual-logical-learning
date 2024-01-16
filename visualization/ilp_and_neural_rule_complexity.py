import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_ilp_neural_data
from visualization.vis_util import make_1_line_im, make_1_im_legend


def rule_complexity_plot(outpath, vis='Trains', im_count=1000):
    labelsize, fontsize = 30, 35
    use_materials = False
    ilp_stats_path = f'{outpath}/ilp/stats'
    neuro_symbolic_stats_path = f'{outpath}/neuro-symbolic/stats'
    alpha_ilp = f'{outpath}/neuro-symbolic/alphailp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    fig_path = f'{outpath}/model_comparison/rule_complexity'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_symbolic_stats_path, alpha_ilp, vis)
    models = neural_models + neuro_symbolic_models + ilp_models
    data = data.loc[data['image noise'] == 0].loc[data['label noise'] == 0].loc[data['training samples'] == im_count].loc[data['Train length'] == '2-4']


    scenes = data['scene'].unique()
    rules = data['rule'].unique()
    rules = ['theoryx', 'numerical', 'complex']

    materials_s = ["///", "//", '/', '\\', '\\\\', 'x', '.', 'o', '+', 'O', '*'] if use_materials else \
        ["//", '\\\\', 'x', '+', "///", '/', '\\', 'o', 'O', '.', '*']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    # colors_s = sns.color_palette('dark')[:len(models)]
    colors_s = list(map(sns.color_palette('Blues').__getitem__, [1, 3, 5]))
    colors_s += list(map(sns.color_palette('Reds').__getitem__, [1, 3, 5]))
    colors_s += list(map(sns.color_palette('Greens').__getitem__, [1, 5]))
    colors = {m: colors_s[n] for n, m in enumerate(models)}

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(26, 3))
    gs = fig.add_gridspec(1, 3, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    sns.set_theme(style="whitegrid")
    for col, rule in enumerate(rules):
        ax = axes[col]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        ax.axvline(x=5.5, color='black', linestyle='--', linewidth=2)

        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        run, y_pos = [], []
        for model in models:
            data_temp = data_t.loc[data['Models'] == model]
            run += ['*'] if len(data_temp) < 5 else ['']
            y_pos += [40 if np.isnan(data_temp['Validation acc'].mean()) else data_temp['Validation acc'].mean()]
            if data_temp.empty:
                continue
            if use_materials:
                sns.barplot(x='Models', y='Validation acc', hue='training samples', data=data_temp, edgecolor='black',
                            palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models)
            else:
                sns.barplot(x='Models', y='Validation acc', data=data_temp, color=colors[model], edgecolor='black',
                            # palette="dark",
                            alpha=.7, ax=ax, orient='v', order=models)
        for container in ax.containers:
            ax.bar_label(container, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
        if use_materials:
            ax.get_legend().remove()
        ax.set_ylim([50, 120])
        ax.get_xaxis().set_visible(False)
        if col != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=labelsize)
        x_pos = sorted(list(set([p.get_x() + p.get_width() / 2 for p in ax.patches])))
        for x, y, r in zip(x_pos, y_pos, run):
            ax.text(x, y + 10, r, fontsize=labelsize, ha='center')

    make_1_im_legend(fig, models, colors, 'Models', [], mt, None,
                     labelsize, legend_h_offset=-.35, legend_v_offset=0, ncols=4)

    os.makedirs(fig_path, exist_ok=True)
    pth = fig_path + f'/rule_complexity.png' if im_count == 1000 else fig_path + f'/rule_complexity_{im_count}_sample.png'
    plt.savefig(pth, bbox_inches='tight', dpi=400)

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
