import glob
import json
from itertools import product

import matplotlib.colors as mcolors
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from numpy import arange
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from torch.utils.data import random_split
import seaborn as sns
import matplotlib.lines as mlines
from util import *

from visualization.vis_model_comparison import get_cv_data


def plot_sinlge_box(rule, vis, out_path, y_val='direction'):
    get_cv_data(f'{out_path}/', y_val)

    with open(out_path + '/label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['visualization'] == vis].loc[data['rule'] == rule]
        data['Validation acc'] = data['Validation acc'].apply(lambda x: x / 100)
        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        visuals = data['visualization'].unique()
        rules = data['rule'].unique()
        noise = data['noise'].unique()
        model_names = data['Methods'].unique()
        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {'SimpleObjects': 'X', 'Trains': 'o', }
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}
    out_path = f'{out_path}/single_evaluation'

    # print(tabulate(data))
    # print(tabulate(data.loc[data['epoch'] == 24].loc[data['Methods'] == 'resnet18'].loc[data['rule'] == 'numerical'].loc[data['visualization'] == 'Trains'], headers='keys', tablefmt='psql'))
    # data = data.loc[data['epoch'] == 24].loc[data['Methods'] == 'resnet18'].loc[data['rule'] == 'numerical']
    # plot over count
    # rules = ['theoryx']
    fig = plt.figure()
    gs = fig.add_gridspec(len(rules), hspace=0)
    axes = gs.subplots(sharex=True, sharey=True)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    # fig, axes = plt.subplots(len(model_names))
    # for model_name, ax in zip(model_names, axes):
    #     data_t = data.loc[data['epoch'] == 24].loc[data['Methods'] == model_name]
    sns.set_theme(style="whitegrid")
    for rule, ax in zip(rules, axes):
        ax.grid(axis='x')
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        data_t = data.loc[data['epoch'] == 24].loc[data['rule'] == rule].loc[data['noise'] == 0]
        # sns.violinplot(y='Validation acc', x='Methods', hue='number of images', data=data_t,
        #                inner="point", linewidth=0.5, dodge=False, palette="pastel", saturation=.5, scale='width',
        #                ax=ax, error='sd', orient='v'
        #                )
        sns.barplot(y='Validation acc', x='Methods', hue='number of images', data=data_t,
                    palette="dark", alpha=.6, ax=ax, orient='v'
                    )
        for count, vis in product(im_count, visuals):
            data_tmp = data_t.loc[data_t['number of images'] == count].loc[data_t['visualization'] == vis]
            # for count in im_count:
            #     data_tmp = data_t.loc[data_t['number of images'] == count]
            #     print(tabulate(data_tmp == data.loc[data['epoch'] == 24].loc[data['Methods'] == 'resnet18'].loc[data['visualization'] == vis].loc[data['number of images'] == count].loc[data['epoch'] == 24], headers='keys', tablefmt='psql'))
            # Show each observation with a scatterplot
            # sns.stripplot(x='Validation acc', y='Methods',
            #               hue='visualization',
            #               hue_order=['SimpleObjects', 'Trains'],
            #               data=data_tmp,
            #               # dodge=True,
            #               alpha=.25,
            #               zorder=1,
            #               jitter=False,
            #               marker=markers[vis],
            #               palette=[colors[count], colors[count]],
            #               ax=ax
            #               )

        # Show the conditional means, aligning each pointplot in the
        # center of the strips by adjusting the width allotted to each
        # category (.8 by default) by the number of hue levels

        # sns.pointplot(x='Validation acc', y='Methods', hue='number of images', data=data_t,
        #               dodge=False,
        #               join=False,
        #               # palette="dark",
        #               markers="d",
        #               scale=.75,
        #               errorbar=None,
        #               ax=ax
        #               )
    # plt.title('Comparison of Supervised learning methods')
    # Improve the legend
    handles, labels = axes[-1].get_legend_handles_labels()
    length = len(handles) // 2
    length = 0
    # simple = mlines.Line2D([], [], color='grey', marker='X', linestyle='None', markersize=5)
    # simple_lab = 'Simple'
    # trains = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5)
    # trains_lab = 'Train'
    # mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
    # mean_lab = 'Mean'
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    for c, ax in enumerate(axes):
        #     ax.get_legend().remove()
        ax.set_ylim([0.5, 1])
    #     ax.set_ylabel(model_names[c])
    #     ax.set_ylabel(rules[c])

    axes[-1].legend(
        # [simple, trains, mean] +
        color_markers,
        # [simple_lab, trains_lab, mean_lab] +
        [str(i) for i in im_count],
        title="training samples",
        loc='lower right',
        # bbox_to_anchor=(1.2, 0),
        # frameon=False,
        handletextpad=0,
        # ncol=3
    )

    # axes[-1].set_ylabel(f"Rule-based learning problem")
    # axes[-1].set_xlabel("Validation accuracy")

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/{vis}_{rule}_violin_neural_lr_mean_variance.png', bbox_inches='tight', dpi=400)

    plt.close()
    # over epoch
    # plot_label_acc_over_epochs(data, out_path)
    # plot_label_acum_acc_over_epoch(data_acum_acc, out_path)

    # plot table
    # csv_to_tex_table(out_path + 'mean_variance_comparison.csv')

    # transfer classification comparison
    # transfer_train_comparison(out_path)
    # csv_to_tex_table(out_path + 'mean_variance_transfer_classification.csv')


from itertools import product

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import matplotlib.lines as mlines
from util import *

from visualization.vis_model_comparison import get_cv_data


def plot_multi_box(rule, visuals, out_path, y_val='direction'):
    get_cv_data(f'{out_path}/', y_val)

    with open(out_path + '/label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['rule'] == rule]

        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        rules = data['rule'].unique()
        noise = data['noise'].unique()
        model_names = data['Methods'].unique()
        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {'SimpleObjects': 'X', 'Trains': 'o', }
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}
    out_path = f'{out_path}/single_evaluation/multibox'

    fig = plt.figure()
    sns.set_theme(style="whitegrid")

    gs = fig.add_gridspec(1, 2, right=1.8)
    axes = gs.subplots(sharex=True, sharey=True)
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

        data_t = data.loc[data['epoch'] == 24].loc[data['visualization'] == vis].loc[data['noise'] == 0]
        # sns.violinplot(y='Validation acc', x='Methods', hue='number of images', data=data_t,
        #                inner="point", linewidth=0.5, dodge=False, palette="pastel", saturation=.5, scale='width',
        #                ax=ax, error='sd', orient='v'
        #                )
        sns.barplot(y='Validation acc', x='Methods', hue='number of images', data=data_t,
                    palette="dark", alpha=.6, ax=ax, orient='v'
                    )
        ax.get_legend().remove()
        ax.set_ylim([0.5, 1])
        if c > 0:
            ax.set_ylabel("")
        ax.set_xlabel("")

        # ax.get_yaxis().set_visible(False)

    # plt.title('Comparison of Supervised learning methods')
    # Improve the legend
    handles, labels = axes[-1].get_legend_handles_labels()
    length = len(handles) // 2
    length = 0
    # simple = mlines.Line2D([], [], color='grey', marker='X', linestyle='None', markersize=5)
    # simple_lab = 'Simple'
    # trains = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5)
    # trains_lab = 'Train'
    # mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
    # mean_lab = 'Mean'
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]


    axes[0].legend(
        # [simple, trains, mean] +
        color_markers,
        # [simple_lab, trains_lab, mean_lab] +
        [str(i) for i in im_count],
        title="number of training samples",
        loc='lower left',
        bbox_to_anchor=(0, -.3),
        frameon=True,
        handletextpad=0,
        ncol=3
    )

    # axes[-1].set_ylabel(f"Rule-based learning problem")
    # axes[-1].set_xlabel("Validation accuracy")

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/{rule}_violin_neural_lr_mean_variance.png', bbox_inches='tight', dpi=400)

    plt.close()
    # over epoch
    # plot_label_acc_over_epochs(data, out_path)
    # plot_label_acum_acc_over_epoch(data_acum_acc, out_path)

    # plot table
    # csv_to_tex_table(out_path + 'mean_variance_comparison.csv')

    # transfer classification comparison
    # transfer_train_comparison(out_path)
    # csv_to_tex_table(out_path + 'mean_variance_transfer_classification.csv')
