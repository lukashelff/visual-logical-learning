import glob
import os
from itertools import product
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
from matplotlib import pyplot as plt


def plot_ilp_bar(noise=0):
    ilp_stats_path = f'output/ilp/stats'
    ilp_vis_path = f'output/ilp/vis'
    noise_tag = '' if noise == 0 else f'_noise_{noise}'

    dirs = glob.glob(ilp_stats_path + '/*.csv')
    data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            data.append(pd.read_csv(f))
    data = pd.concat(data, ignore_index=True)

    rules = data['rule'].unique()
    models = data['Methods'].unique()
    rules = ['theoryx', 'numerical', 'complex', ]

    # data = pd.concat(data, ignore_index=True)
    data = data.loc[data['noise'] == noise]
    # data = data.loc[data['rules'] == 'theoryx']
    data['Validation acc'] = data['Validation acc'] * 100

    im_count = sorted(data['training samples'].unique())
    # print(tabulate(data))

    sns.set_theme(style="whitegrid")
    colors_s = sns.color_palette()[:len(im_count) + 1]
    colors = {count: colors_s[n] for n, count in enumerate(im_count)}
    materials_s = ["///", '/']
    mt = {model: materials_s[n] for n, model in enumerate(models)}

    fig = plt.figure(figsize=(7,6))
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.15)

    axes = gs.subplots(sharex=True, sharey=False)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for c, rule in enumerate(rules):
        ax = axes[c // 2, c % 2]
        ax.title.set_text(rule.title())
        ax.grid(axis='x')
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            sns.barplot(data=data_temp, x='Methods', y='Validation acc', hue='training samples', ax=ax, alpha=.7,
                        palette='dark', order=models, hatch=mt[model], orient='v')
        ax.get_legend().remove()
        ax.set_ylim([50, 100])
        ax.get_xaxis().set_visible(False)
        if c % 2:
            # ax.get_yaxis().set_visible(False)
            ax.set_ylabel('')
            ax.set_yticklabels([''] * 6)
        else:
            ax.set_ylabel('Accuracy')

    # Improve the legend
    axes[1, 1].set_axis_off()
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    plt.rcParams.update({'hatch.color': 'black'})
    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    leg = fig.legend(
        white + color_markers + white + handels,
        ['Training samples:'] + im_count + ['Models:'] + [m.title() for m in models],
        loc='lower left',
        bbox_to_anchor=(.515, 0.292),
        frameon=True,
        handletextpad=0,
        ncol=2, handleheight=1.2, handlelength=2.2
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    # for vpack in leg._legend_handle_box.get_children():
    #     for hpack in vpack.get_children()[:1]:
    #         hpack.get_children()[0].set_width(0)

    os.makedirs(ilp_vis_path, exist_ok=True)
    plt.savefig(ilp_vis_path + f'/ilp_bar{noise_tag}.png', bbox_inches='tight', dpi=400)

    plt.close()
