import glob
import os

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def elementary_vs_realistic(neural_path, ilp_pth, tr_samples=1000):
    ilp_stats_path = f'{ilp_pth}/stats'

    dirs = glob.glob(ilp_stats_path + '/*.csv')
    ilp_data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            ilp_data.append(pd.read_csv(f))
    ilp_data = pd.concat(ilp_data, ignore_index=True)
    ilp_data['visualization'] = 'Trains'
    ilp_models = sorted(ilp_data['Methods'].unique())

    with open(neural_path + '/label_acc_over_epoch.csv', 'r') as f:
        neur_data = pd.read_csv(f)
        neur_data = neur_data.loc[neur_data['epoch'] == 24]
    neur_data = neur_data.rename({'number of images': 'training samples'}, axis='columns')
    neural_models = sorted(neur_data['Methods'].unique())

    data = pd.concat([ilp_data, neur_data], ignore_index=True)
    data = data.loc[data['training samples'] == tr_samples].loc[data['noise'] == 0]

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    rules = data['rule'].unique()
    noise = data['noise'].unique()
    models = np.append(neural_models, ilp_models)
    visualizations = list(data['visualization'].unique())
    # models = np.append(neural_models, ilp_models)
    colors_s = sns.color_palette()[:len(visualizations) + 1]
    colors = {vis: colors_s[n] for n, vis in enumerate(visualizations)}
    rules = ['theoryx', 'numerical', 'complex']

    out_path = f'{neural_path}/elementary_vs_realistic'
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
        ax.title.set_text(rule.title())
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            sns.barplot(x='Methods', y='Validation acc', hue='visualization', hue_order=visualizations,
                        data=data_temp, palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
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
                     visualizations]
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    plt.rcParams.update({'hatch.color': 'black'})

    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    leg = fig.legend(
        white + color_markers + white * 4 + handels,
        ['Visualizations:'] + visualizations + [''] * 3 + ['Models:'] + [m.title() for m in models],
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
    plt.savefig(out_path + f'/elementary_vs_realistic_{tr_samples}_samples.png', bbox_inches='tight', dpi=400)

    plt.close()
