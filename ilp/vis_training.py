import glob
import os
from itertools import product

import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
import tabulate
from matplotlib import pyplot as plt

def plot_ilp_training(noise=0):
    ilp_stats_path = f'output/ilp/stats'
    ilp_vis_path = f'output/ilp/vis'
    noise_tag = '' if noise == 0 else f'_noise_{noise}'

    dirs = glob.glob(ilp_stats_path + '/*.csv')
    data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            data.append(pd.read_csv(f))
    data = pd.concat(data, ignore_index=True)

    # with open(ilp_stats_path + f'/popper_stats{noise_tag}.csv', 'r') as pop_stat, open(
    #         ilp_stats_path + f'/aleph_stats_{noise_tag}.csv', 'r') as aleph_stat:
    #     a, p = pd.read_csv(pop_stat), pd.read_csv(aleph_stat)
    rules = data['rule'].unique()
    methods = data['Methods'].unique()
    rules = ['numerical', 'theoryx', 'complex', ]

    # data = pd.concat(data, ignore_index=True)
    data = data.loc[data['noise'] == noise]
    # data = data.loc[data['rules'] == 'theoryx']
    data['Validation acc'] = data['Validation acc'] * 100

    im_count = sorted(data['training samples'].unique())
    # print(tabulate(data))

    sns.set_theme(style="whitegrid")
    f, axes = plt.subplots(2, 3, figsize=(12, 12))
    colors_s = sns.color_palette()[:len(im_count) + 1]
    colors = {count: colors_s[n] for n, count in enumerate(im_count)}

    for (method, rule), ax in zip(product(methods, rules), axes.flatten()):

        data_t = data.loc[data['rule'] == rule].loc[data['Methods'] == method]
        sns.barplot(data=data_t, x='Methods', y='Validation acc', hue='training samples', ax=ax, alpha=.6)

        # ax.grid(axis='x', linestyle='solid', color='gray')
        # ax.tick_params(bottom=False, left=False)
        # for spine in ax.spines.values():
        #     spine.set_edgecolor('gray')
        # data_t = data.loc[data['rule'] == rule].sort_values(by=['Methods'], ascending=True)
        # # sns.violinplot(x='Validation acc', y='rule', hue='number of images', data=data_t,
        # #                inner="quart", linewidth=0.5, dodge=False, palette="pastel", saturation=.2, scale='width',
        # #                ax=ax
        # #                )
        # for count in im_count:
        #     data_tmp = data_t.loc[data_t['training samples'] == count]
        #     # Show each observation with a scatterplot
        #     sns.stripplot(x='Validation acc', y='Methods', hue='training samples',
        #                   data=data_tmp,
        #                   dodge=True,
        #                   alpha=.25,
        #                   zorder=1,
        #                   jitter=False,
        #                   palette=[colors[count]],
        #                   ax=ax
        #                   )
        #
        # # Show the conditional means, aligning each pointplot in the
        # # center of the strips by adjusting the width allotted to each
        # # category (.8 by default) by the number of hue levels
        # sns.pointplot(x='Validation acc', y='Methods', hue='training samples', data=data_t,
        #               dodge=False,
        #               join=False,
        #               # palette="dark",
        #               markers="d",
        #               scale=.75,
        #               errorbar=None,
        #               ax=ax
        #               )
    # Improve the legend
    trains = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5)
    trains_lab = 'Val acc'
    mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
    mean_lab = 'Mean'
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    x_lab = list(product(methods, rules))
    for c, ax in enumerate(axes.flatten()):
        ax.get_legend().remove()
        ax.set_ylim([50, 100])
        if c % 3 == 0:
            ax.set_ylabel(x_lab[c][0])
        else:
            ax.get_yaxis().set_visible(False)
        if c < 3:
            ax.get_xaxis().set_visible(False)
        else:
            ax.get_xaxis().set_ticks([])
            ax.set_xlabel(x_lab[c][1])


    os.makedirs(ilp_vis_path, exist_ok=True)
    plt.savefig(ilp_vis_path + f'/ilp_bar{noise_tag}.png', bbox_inches='tight', dpi=400)

    plt.close()