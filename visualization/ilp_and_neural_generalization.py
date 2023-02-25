import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def vis_generalization_ilp_and_neural(neural_path, ilp_pth, vis='Trains', min_cars=7, max_cars=7):

    ilp_stats_path = f'{ilp_pth}/stats'

    dirs = glob.glob(ilp_stats_path + '/*.csv')
    ilp_data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            ilp_data.append(pd.read_csv(f))
    ilp_data = pd.concat(ilp_data, ignore_index=True)
    ilp_data['Train length'] = '2-4'
    ilp_data = ilp_data.loc[ilp_data['training samples'] == 10000].loc[ilp_data['noise'] == 0]
    ilp_models = sorted(ilp_data['Methods'].unique())

    with open(neural_path + f'/cnn_generalization_{min_cars}_{max_cars}.csv', 'r') as f:
        data_gen = pd.read_csv(f)
    data_gen = data_gen.rename({'number of images': 'training samples'}, axis='columns')
    data_gen['Train length'] = '7'

    with open(neural_path + '/label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['epoch'] == 24].loc[data['visualization'] == vis].loc[data['number of images'] == 10000].loc[data['noise'] == 0]
    data = data.rename({'number of images': 'training samples'}, axis='columns')
    data['Train length'] = '2-4'
    neural_models = sorted(data['Methods'].unique())

    # data = pd.concat([ilp_data, data, data_gen])
    data = pd.concat([ilp_data, data, data_gen])

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    datasets = list(data['Train length'].unique())
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    models = np.append(neural_models, ilp_models)
    colors_s = sns.color_palette()[:len(datasets)]
    colors = {datasets[0]: colors_s[0], datasets[1]: colors_s[1]}
    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{neural_path}/ilp_and_neural_generalization'
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
        ax.title.set_text(rule.title())
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            sns.barplot(x='Methods', y='Validation acc', hue='Train length', data=data_temp,
                        palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                        )
        ax.get_legend().remove()
        ax.set_ylim([0.5, 1])
        ax.get_xaxis().set_visible(False)
        if c % 2:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            # ax.set_yticklabels([''] * 9)
        else:
            ax.set_ylabel('Accuracy')


    axes[1, 1].set_axis_off()
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     datasets]
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    plt.rcParams.update({'hatch.color': 'black'})

    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    leg = fig.legend(
        white + color_markers + white*4 + handels,
        ['Train length'] + datasets + ['']*3 + ['Models:'] + [m.title() for m in models],
        loc='lower left',
        bbox_to_anchor=(.515, 0.21),
        frameon=True,
        handletextpad=0,
        ncol=2, handleheight=1.2, handlelength=2.2
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/generalization_task.png', bbox_inches='tight', dpi=400)

    plt.close()
