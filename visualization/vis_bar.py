import os
from itertools import product, chain
import matplotlib.patches as mpatches

import matplotlib.lines as mlines
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_cv_data


def plot_sinlge_box(rule, vis, out_path):
    with open(out_path + '/label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['visualization'] == vis].loc[data['rule'] == rule].loc[data['epoch'] == 24].loc[
            data['noise'] == 0]
        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        visuals = data['visualization'].unique()
        rules = data['rule'].unique()
        noise = data['noise'].unique()
        models = sorted(data['Methods'].unique())
        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {'SimpleObjects': 'X', 'Trains': 'o', }
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}
    out_path = f'{out_path}/single_evaluation/simple/'
    materials_s = ["///", '/', '\\']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    data = data.loc[data['rule'] == rule].loc[data['visualization'] == vis]
    sns.set_theme(style="whitegrid")
    ax = plt.gca()

    for spine in ax.spines.values():
        spine.set_edgecolor('gray')

    for model in models:
        data_temp = data.loc[data['Methods'] == model]
        sns.barplot(y='Validation acc', x='Methods', hue='number of images', data=data_temp,
                    palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models)

    ax.grid(axis='x')
    ax.tick_params(bottom=False, left=False)
    ax.set_ylim([0.5, 1])
    ax.get_xaxis().set_visible(False)
    ax.set_ylabel('Accuracy')
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    white = mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)
    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    ax.legend(

        [white, white] + list(chain.from_iterable(zip(color_markers, handels))),
        ['Training samples:', 'Models:'] + list(chain.from_iterable(zip(im_count, models))),

        loc='lower center',
        bbox_to_anchor=(.505, -.2),
        frameon=True,
        handletextpad=0,
        ncol=4, handleheight=1.2, handlelength=2.2
    )

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/{vis}_{rule}_bar_neural_lr_mean_variance.png', bbox_inches='tight', dpi=400)
    plt.close()


def plot_multi_box(rule, visuals, out_path):
    with open(out_path + '/label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['rule'] == rule].loc[data['epoch'] == 24].loc[data['noise'] == 0]

        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        rules = data['rule'].unique()
        noise = data['noise'].unique()
        models = sorted(data['Methods'].unique())
        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {'SimpleObjects': 'X', 'Trains': 'o', }
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}
    out_path = f'{out_path}/single_evaluation/multibox'
    materials_s = ["///", '/', '\\']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    fig = plt.figure()

    gs = fig.add_gridspec(1, 2, right=1.5, wspace=.05)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    # fig, axes = plt.subplots(len(model_names))
    # for model_name, ax in zip(model_names, axes):
    #     data_t = data.loc[data['epoch'] == 24].loc[data['Methods'] == model_name]
    sns.set_theme(style="whitegrid")
    for c, (vis, ax) in enumerate(zip(visuals, axes)):
        ax.grid(axis='x')
        ax.tick_params(bottom=False, left=False)
        ax.title.set_text(vis)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['visualization'] == vis]
        for model in models:
            data_temp = data_t.loc[data['Methods'] == model]
            sns.barplot(y='Validation acc', x='Methods', hue='number of images', data=data_temp,
                        palette="dark", alpha=.7, ax=ax, orient='v', hatch=mt[model], order=models
                        )
        ax.get_legend().remove()
        ax.set_ylim([0.5, 1])
        if c > 0:
            ax.set_ylabel("")
        else:
            ax.set_ylabel('Accuracy')

        ax.get_xaxis().set_visible(False)

    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    white = mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)
    plt.rcParams.update({'hatch.color': 'black'})

    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    leg = ax.legend(
        [white, white] + list(chain.from_iterable(zip(color_markers, handels))),
        ['Training samples:', 'Models:'] + list(chain.from_iterable(zip(im_count, models))),
        loc='lower center',
        bbox_to_anchor=(-.1, -.18),
        frameon=True,
        handletextpad=0,
        ncol=4, handleheight=1.2, handlelength=2.2
    )
    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/{rule}_bar_neural_lr_mean_variance.png', bbox_inches='tight', dpi=400)

    plt.close()


def plot_rules_bar(out_path, vis='Trains'):
    with open(out_path + '/label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['epoch'] == 24].loc[data['noise'] == 0].loc[data['visualization'] == vis]
        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        rules = data['rule'].unique()
        noise = data['noise'].unique()
        models = sorted(data['Methods'].unique())
        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {'SimpleObjects': 'X', 'Trains': 'o', }
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}
    rules = ['theoryx', 'numerical', 'complex', ]

    out_path = f'{out_path}/single_evaluation/multibox'
    materials_s = ["///", '/', '\\']
    mt = {model: materials_s[n] for n, model in enumerate(models)}
    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    # fig, axes = plt.subplots(len(model_names))
    # for model_name, ax in zip(model_names, axes):
    #     data_t = data.loc[data['epoch'] == 24].loc[data['Methods'] == model_name]
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
            sns.barplot(x='Methods', y='Validation acc', hue='number of images', data=data_temp,
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

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/{vis}_bar_neural_rules.png', bbox_inches='tight', dpi=400)

    plt.close()


def plot_neural_noise_as_bars(out_path, y_val='direction'):
    # get_cv_data(f'{out_path}/', y_val)

    with open(out_path + '/label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['epoch'] == 24].loc[data['visualization'] == 'Trains']

        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        rules = data['rule'].unique()
        noise = sorted(data['noise'].unique())
        model_names = sorted(data['Methods'].unique())
        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {'SimpleObjects': 'X', 'Trains': 'o', }
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}

    fig = plt.figure()
    gs = fig.add_gridspec(len(model_names), len(rules), wspace=0, right=1.5)
    sns.set_theme(style="whitegrid")

    axes = gs.subplots(sharex=True, sharey=False)
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    for c, ((ns, rule), ax) in enumerate(zip(product(noise, rules), axes.flatten())):
        ax.grid(axis='y')
        ax.tick_params(bottom=False, left=False)
        ax.set_ylim([0.5, 1])
        # if c < 6:
        #     ax.get_xaxis().set_visible(False)
        # else:
        #     ax.set_xlabel('')
        if c % 3 == 0:
            ax.set_ylabel(f'{noise[c // 3] * 100}% noise')
            print('aaa')
        else:
            ax.get_yaxis().set_visible(False)

            # ax.set_ylabel('')

        if c < 3:
            ax.title.set_text(rule)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        data_t = data.loc[data['rule'] == rule].loc[data['noise'] == ns]
        sns.barplot(y='Validation acc', x='Methods', hue='number of images', data=data_t,
                    palette="dark", alpha=.7, ax=ax, orient='v'
                    )
        ax.get_legend().remove()
        ax.get_xaxis().set_visible(False)

    # plt.title('Comparison of Supervised learning methods')
    # Improve the legend

    # simple = mlines.Line2D([], [], color='grey', marker='X', linestyle='None', markersize=5)
    # simple_lab = 'Simple'
    # trains = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5)
    # trains_lab = 'Train'
    # mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
    # mean_lab = 'Mean'
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    # for c, ax in enumerate(axes.flatten()):
    #     # ax.get_legend().remove()
    #     ax.set_xlim([0.5, 1])
    #     if c % 2 == 0:
    #         ax.set_ylabel(model_names[c % 3])
    #     else:
    #         ax.get_yaxis().set_visible(False)

    axes[2, 1].legend(
        # [simple, trains, mean] +
        color_markers,
        # [simple_lab, trains_lab, mean_lab] +
        [f'{i * 100}%' for i in noise],
        title="Noise on labels",
        loc='lower center', bbox_to_anchor=(-.04, -.85), frameon=True,
        ncol=3
    )

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/neural_on_noise.png', bbox_inches='tight', dpi=400)

    plt.close()
