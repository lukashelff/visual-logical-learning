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
from tabulate import tabulate

from visualization.data_handler import get_cv_data


def plot_neural_noise(out_path, y_val='direction'):
    _out_path = f'{out_path}/'
    get_cv_data(_out_path, y_val)

    with open(_out_path + 'label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['epoch'] == 24].loc[data['visualization'] == 'Trains']
        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        visuals = data['visualization'].unique()
        rules = data['rule'].unique()
        noise = data['noise'].unique()
        models = data['Methods'].unique()
        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {f'{models[0]}': 'X', f'{models[1]}': 'o', f'{models[2]}': '>'}
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}
    # print(tabulate(data))
    # print(tabulate(data.loc[data['epoch'] == 24].loc[data['Methods'] == 'resnet18'].loc[data['rule'] == 'numerical'].loc[data['visualization'] == 'Trains'], headers='keys', tablefmt='psql'))
    # data = data.loc[data['epoch'] == 24].loc[data['Methods'] == 'resnet18'].loc[data['rule'] == 'numerical']
    # plot over count
    rules = ['theoryx', 'numerical', 'complex']
    data['noise'] = (data['noise'] * 100).astype("int").astype("string") + '%'
    fig = plt.figure(figsize=(17, 7))
    gs = fig.add_gridspec(len(rules), len(models), hspace=0)
    axes = gs.subplots(sharex=True, sharey=False)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    # fig, axes = plt.subplots(len(model_names))
    # for model_name, ax in zip(model_names, axes):
    #     data_t = data.loc[data['epoch'] == 24].loc[data['Methods'] == model_name]
    for (model, rule), ax in zip(product(models, rules), axes.flatten()):
        ax.grid(axis='x', linestyle='solid', color='gray')
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        data_t = data.loc[data['rule'] == rule].loc[data['Methods'] == model].sort_values(by=['noise'], ascending=True)

        # sns.violinplot(x='Validation acc', y='rule', hue='number of images', data=data_t,
        #                inner="quart", linewidth=0.5, dodge=False, palette="pastel", saturation=.2, scale='width',
        #                ax=ax
        #                )
        for count in im_count:
            data_tmp = data_t.loc[data_t['number of images'] == count]
            # for count in im_count:
            #     data_tmp = data_t.loc[data_t['number of images'] == count]
            #     print(tabulate(data_tmp == data.loc[data['epoch'] == 24].loc[data['Methods'] == 'resnet18'].loc[data['visualization'] == vis].loc[data['number of images'] == count].loc[data['epoch'] == 24], headers='keys', tablefmt='psql'))
            # Show each observation with a scatterplot
            sns.stripplot(x='Validation acc', y='noise',
                          hue='number of images',
                          # hue_order=['SimpleObjects', 'Trains'],
                          data=data_tmp,
                          dodge=False,
                          alpha=.25,
                          zorder=1,
                          jitter=False,
                          marker=markers[model],
                          palette=[colors[count]],
                          ax=ax
                          )

        # Show the conditional means, aligning each pointplot in the
        # center of the strips by adjusting the width allotted to each
        # category (.8 by default) by the number of hue levels

        sns.pointplot(x='Validation acc', y='noise', hue='number of images', data=data_t,
                      dodge=False,
                      join=False,
                      # palette="dark",
                      markers="d",
                      scale=.75,
                      errorbar=None,
                      ax=ax
                      )
    # plt.title('Comparison of Supervised learning methods')
    # Improve the legend
    handles, labels = axes[2, 0].get_legend_handles_labels()
    length = len(handles) // 2
    # length = 0
    white = mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)
    h1 = mlines.Line2D([], [], color='grey', marker='X', linestyle='None', markersize=5)
    lab1 = models[0]
    h2 = mlines.Line2D([], [], color='grey', marker='o', linestyle='None', markersize=5)
    lab2 = models[1]
    h3 = mlines.Line2D([], [], color='grey', marker='>', linestyle='None', markersize=5)
    lab3 = models[2]
    mean = mlines.Line2D([], [], color='grey', marker='d', linestyle='None', markersize=5)
    mean_lab = 'Mean accuracy'
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    for c, ax in enumerate(axes.flatten()):
        ax.get_legend().remove()
        ax.set_xlim([0.5, 1])
        if c % 3 == 0:
            ax.set_ylabel('noise')
            # ax.set_ylabel(models[c // 3])
            # ax.set_ylabel(rules[c % 3])
        else:
            # ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
        if c < 3:
            ax.title.set_text(rules[c])
    # labels, handels = [str(i) for i in im_count], []
    # for i in range(3):
    #     labels
    axes[2, 1].legend(
        [white, white, color_markers[0], h1, color_markers[1], h2, color_markers[2], h3,
                                                   mean],
        ['Training samples:', 'Models:'] + [im_count[0], lab1, im_count[1], lab2, im_count[2], lab3, mean_lab],
        loc='lower center', bbox_to_anchor=(0.5, -.57), frameon=True,
        handletextpad=0, ncol=5)

    # axes[-1].set_ylabel(f"Rule-based learning problem")
    # axes[2, 0].set_xlabel("Validation accuracy")

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/neural_on_noise_v2.png', bbox_inches='tight', dpi=400)

    plt.close()
    # over epoch
    # plot_label_acc_over_epochs(data, out_path)
    # plot_label_acum_acc_over_epoch(data_acum_acc, out_path)

    # plot table
    # csv_to_tex_table(out_path + 'mean_variance_comparison.csv')

    # transfer classification comparison
    # transfer_train_comparison(out_path)
    # csv_to_tex_table(out_path + 'mean_variance_transfer_classification.csv')



