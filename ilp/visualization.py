import glob
import os
from itertools import product
import matplotlib.patches as mpatches

import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec


def plot_ilp_crossval(noise=0):
    ilp_stats_path = f'output/ilp/stats'
    ilp_vis_path = f'output/ilp/vis'
    noise_tag = '' if noise == 0 else f'_noise_{noise}'

    dirs = glob.glob(ilp_stats_path + '/*.csv')
    data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            data.append(pd.read_csv(f))

    # with open(ilp_stats_path + f'/popper_stats{noise_tag}.csv', 'r') as pop_stat, open(
    #         ilp_stats_path + f'/aleph_stats_{noise_tag}.csv', 'r') as aleph_stat:
    #     a, p = pd.read_csv(pop_stat), pd.read_csv(aleph_stat)
    data = pd.concat(data, ignore_index=True)
    data = data.loc[data['noise'] == noise]
    rules = data['rule'].unique()
    model_names = data['Methods'].unique()
    im_count = sorted(data['training samples'].unique())
    # print(tabulate(data))

    fig = plt.figure()
    gs = fig.add_gridspec(len(rules), hspace=0)
    axes = gs.subplots(sharex=True, sharey=True)
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    colors_s = sns.color_palette()[:len(im_count) + 1]
    colors = {count: colors_s[n] for n, count in enumerate(im_count)}
    # colors[count] = colors_s[n]
    # colors = {
    #     10000: colors_s[2],
    #     1000: colors_s[1],
    #     100: colors_s[0]
    # }
    for rule, ax in zip(rules, axes):
        ax.grid(axis='x', linestyle='solid', color='gray')
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule].sort_values(by=['Methods'], ascending=True)
        # sns.violinplot(x='Validation acc', y='rule', hue='number of images', data=data_t,
        #                inner="quart", linewidth=0.5, dodge=False, palette="pastel", saturation=.2, scale='width',
        #                ax=ax
        #                )
        for count in im_count:
            data_tmp = data_t.loc[data_t['training samples'] == count]
            # Show each observation with a scatterplot
            sns.stripplot(x='Validation acc', y='Methods', hue='training samples',
                          data=data_tmp,
                          dodge=True,
                          alpha=.25,
                          zorder=1,
                          jitter=False,
                          palette=[colors[count]],
                          ax=ax
                          )

        # Show the conditional means, aligning each pointplot in the
        # center of the strips by adjusting the width allotted to each
        # category (.8 by default) by the number of hue levels
        sns.pointplot(x='Validation acc', y='Methods', hue='training samples', data=data_t,
                      dodge=False,
                      join=False,
                      # palette="dark",
                      markers="d",
                      scale=.75,
                      errorbar=None,
                      ax=ax
                      )
    # Improve the legend
    trains = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5)
    trains_lab = 'Val acc'
    mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
    mean_lab = 'Mean'
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    for c, ax in enumerate(axes):
        ax.get_legend().remove()
        ax.set_xlim([0.5, 1])
        ax.set_ylabel(rules[c])

    axes[-1].legend(color_markers + [trains, mean],
                    [str(i) for i in im_count] + [trains_lab, mean_lab], title="Training samples",
                    loc='lower center', bbox_to_anchor=(1.2, 0), frameon=False,
                    handletextpad=0, ncol=2)

    # axes[-1].set_ylabel(f"Rule-based learning problem")
    axes[-1].set_xlabel("Validation accuracy")

    os.makedirs(ilp_vis_path, exist_ok=True)
    plt.savefig(ilp_vis_path + f'/ilp_mean_variance{noise_tag}.png', bbox_inches='tight', dpi=400)

    plt.close()


def plot_noise_robustness():
    ilp_stats_path = f'output/ilp/stats'
    ilp_vis_path = f'output/ilp/vis'
    dirs = glob.glob(ilp_stats_path + '/*.csv')
    data = []
    for dir in dirs:
        with open(dir, 'r') as f:
            tmp = pd.read_csv(f)
            if tmp.empty:
                os.remove(dir)
            else:
                data.append(tmp)

    data = pd.concat(data, ignore_index=True)
    # data = (data.loc[data['rule'] == 'numerical']).loc[data['Methods'] == 'popper']
    data['noise'] = (data['noise'] * 100).astype("int").astype("string") + '%'
    rules = data['rule'].unique()
    rules = ['theoryx', 'numerical', 'complex']
    models = data['Methods'].unique()
    im_count = sorted(data['training samples'].unique())

    fig = plt.figure(figsize=(8, 6))
    outer = fig.add_gridspec(2, 2, hspace=.15, wspace=0.1, figure=fig)
    colors_s = sns.color_palette()[:len(im_count) + 1]
    colors = {count: colors_s[n] for n, count in enumerate(im_count)}
    markers = {f'{models[0]}': 'X', f'{models[1]}': 'o'}

    for c, rule in enumerate(rules):
        out = outer[c // 2, c % 2]
        inner = out.subgridspec(ncols=1, nrows=2, hspace=0)
        axes = inner.subplots()
        axes[0].set_title(rule.title())
        # inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[c], wspace=0.1, hspace=0)
        # inner = inner.subplots(sharex=True, sharey=False)
        for j in range(len(models)):
            # ax = plt.Subplot(fig, inner[j,0])
            model, ax = models[j], axes[j]
            # ax.grid(axis='x', linestyle='solid', color='gray')
            ax.tick_params(bottom=False, left=False)
            ax.grid(axis='x')
            for spine in ax.spines.values():
                spine.set_edgecolor('gray')
            data_t = (data.loc[data['rule'] == rule]).loc[data['Methods'] == model].sort_values(by=['noise'],
                                                                                                ascending=True)
            for count in im_count:
                data_tmp = data_t.loc[data_t['training samples'] == count]
                # Show each observation with a scatterplot
                if len(data_tmp) != 0:
                    sns.stripplot(x='Validation acc', y='noise', hue='training samples',
                                  # y_axis=['10%', '0%'],
                                  data=data_tmp,
                                  dodge=False,
                                  alpha=.25,
                                  zorder=1,
                                  size=6,
                                  jitter=False,
                                  marker=markers[model],
                                  palette=[colors[count]],
                                  ax=ax
                                  )

            # Show the conditional means, aligning each pointplot in the
            # center of the strips by adjusting the width allotted to each
            # category (.8 by default) by the number of hue levels
            if len(data_t) != 0:
                sns.pointplot(x='Validation acc', y='noise', hue='training samples', data=data_t,
                              # y_axis=['10%', '0%'],
                              dodge=False,
                              join=False,
                              # palette="dark",
                              markers="d",
                              scale=.75,
                              errorbar=None,
                              errwidth=0,
                              ax=ax
                              )

            ax.get_legend().remove()
            if c % 2:
                ax.get_yaxis().set_visible(False)
            else:
                ax.set_ylabel('Noise')
            ax.set_xlim([.5, 1])
            if j % 2 == 0 or c == 0:
                ax.set_xlabel('')
                ax.set_xticklabels([''] * 6)
            else:
                ax.set_xlabel('Accuracy')

    # Improve the legend
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    popper = mlines.Line2D([], [], color='grey', marker=markers['popper'], linestyle='None', markersize=5)
    aleph = mlines.Line2D([], [], color='grey', marker=markers['aleph'], linestyle='None', markersize=5)
    mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
    mean_lab = 'Mean Accuracy'

    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    plt.rcParams.update({'hatch.color': 'black'})

    leg = fig.legend(
        white + color_markers + white + [aleph, popper, mean],
        ['Training Samples:'] + im_count + ['Models:'] + [m.title() for m in models] + [mean_lab],
        loc='lower left',
        bbox_to_anchor=(.522, 0.26),
        frameon=True,
        handletextpad=0,
        ncol=2, handleheight=1.2, handlelength=2.2
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)

    os.makedirs(ilp_vis_path, exist_ok=True)
    plt.savefig(ilp_vis_path + f'/ilp_on_noisy_data.png', bbox_inches='tight', dpi=400)

    plt.close()
