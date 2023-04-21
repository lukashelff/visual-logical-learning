import glob
import json
from itertools import product, chain

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


def rule_comparison(out_path):
    _out_path = f'{out_path}/'

    with open(_out_path + 'label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        data = data.loc[data['noise'] == 0].loc[data['epoch'] == 24]
        scenes = data['scene'].unique()
        im_count = sorted(data['number of images'].unique())
        visuals = sorted(data['visualization'].unique())
        rules = data['rule'].unique()
        noise = data['noise'].unique()
        models = sorted(data['Methods'].unique())

        colors_s = sns.color_palette()[:len(im_count) + 1]
        markers = {f'{models[0]}': 'X', f'{models[1]}': 'o', f'{models[2]}': 'd'}
        colors = {10000: colors_s[2], 1000: colors_s[1], 100: colors_s[0]}
    rules = ['theoryx', 'numerical', 'complex']

    fig = plt.figure()
    gs = fig.add_gridspec(len(rules), len(visuals), hspace=0, wspace=0.1, right=1.3)
    axes = gs.subplots(sharex=True, sharey=False)
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for (rule, vis), ax in zip(product(rules, visuals), axes.flatten()):
        ax.grid(axis='x', linestyle='solid', color='gray')
        ax.tick_params(bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')

        data_t = data.loc[data['rule'] == rule].loc[data['visualization'] == vis]
        # sns.violinplot(x='Validation acc', y='rule', hue='number of images', data=data_t,
        #                inner="quart", linewidth=0.5, dodge=False, palette="pastel", saturation=.2, scale='width',
        #                ax=ax
        #                )
        for count in im_count:
            data_tmp = data_t.loc[data_t['number of images'] == count]
            for model in models:
                data_tmp2 = data_tmp.loc[data_t['Methods'] == model]

                sns.stripplot(x='Validation acc', y='Methods',
                              hue='number of images',
                              hue_order=im_count,
                              data=data_tmp2,
                              dodge=False,
                              alpha=.25,
                              zorder=1,
                              jitter=False,
                              marker=markers[model],
                              palette=sns.color_palette()[:len(im_count)],
                              ax=ax
                              )

        # Show the conditional means, aligning each pointplot in the
        # center of the strips by adjusting the width allotted to each
        # category (.8 by default) by the number of hue levels
        # for model in models:
        #     data_tmp2 = data_t.loc[data_t['Methods'] == model]
        for model in models:
            data_tm2 = data_t.loc[data_t['Methods'] == model]
            sns.pointplot(x='Validation acc', y='Methods', hue='number of images', data=data_tm2, order=models,
                          dodge=False,
                          join=False,
                          # palette="dark",
                          markers=markers[model],
                          scale=.6,
                          errorbar=None,
                          ax=ax
                          )
        ax.get_legend().remove()
        ax.set_xlim([0.5, 1])
        if vis == visuals[0]:
            ax.set_ylabel(rule)
            ax.get_yaxis().set_ticks([])
        else:
            ax.get_yaxis().set_visible(False)
        if rule == rules[0]:
            ax.title.set_text(vis)


    white = mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)
    handels = [mlines.Line2D([], [], color='grey', marker=markers[m], linestyle='None', markersize=5) for m in models]
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     im_count]
    leg = fig.legend(
        [white, white] + list(chain.from_iterable(zip(color_markers, handels))),
        ['Training samples:', 'Models:'] + list(chain.from_iterable(zip(im_count, models))),
        loc='lower center', bbox_to_anchor=(.72, -.115), frameon=True,
        handletextpad=0, ncol=4)
    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)

    os.makedirs(out_path, exist_ok=True)
    plt.savefig(out_path + f'/neural_lr_mean_variance.png', bbox_inches='tight', dpi=400)

    plt.close()
    # over epoch


def model_scene_imcount_comparison(raw_trains, model_names, y_val, out_path, transfer_eval=False):
    # full_ds = get_datasets(base_scene, raw_trains, 1)
    #
    # label_names = full_ds.labels
    # unique_label_names = list(set(label_names))
    # label_classes = full_ds.label_classes

    data = pd.DataFrame(
        columns=['Methods', 'number of images', 'cv iteration', 'Validation acc', 'epoch', 'scene', 'label'])
    data_acum_acc = pd.DataFrame(
        columns=['Methods', 'number of images', 'cv iteration', 'Validation acc', 'Train acc', 'epoch', 'scene'])
    data_ev = pd.DataFrame(
        columns=['Methods', 'number of images', 'scene', 'mean', 'variance', 'std'])
    data_transfer_classification = pd.DataFrame(
        columns=['Methods', 'number of images', 'cv iteration', 'Validation acc', 'scene'])
    data_transfer_classification_ev = pd.DataFrame(
        columns=['Methods', 'number of images', 'scene', 'mean', 'variance', 'std'])
    for model_name in model_names:

        _out_path = f'output/models/{model_name}/{y_val}_classification/'
        scenes = os.listdir(_out_path)
        if transfer_eval:
            with open(_out_path + f'/base_scene/transfer_classification_cv.csv', 'r') as f:
                _df = pd.read_csv(f)
                data_transfer_classification = pd.concat([data_transfer_classification, _df], ignore_index=True)
            with open(_out_path + f'/base_scene/transfer_classification.csv', 'r') as f:
                _df = pd.read_csv(f)
                _df = _df.rename(columns={'methods': 'Methods', 'scenes': 'scene'})
                data_transfer_classification_ev = pd.concat([data_transfer_classification_ev, _df], ignore_index=True)
        for scene in scenes:
            _out_path = f'output/models/{model_name}/{y_val}_classification/{scene}/cv/'
            configs = os.listdir(_out_path)
            configs.insert(0, configs.pop())
            for config in configs:
                conf = config.split('_')
                imcount = int(conf[1])
                dir = _out_path + config
                cv_paths = glob.glob(dir + '/*/metrics.json')

                final_acc = []
                for iteration, path in enumerate(cv_paths):
                    with open(path, 'r') as fp:
                        statistics = json.load(fp)
                    epoch_label_accs = statistics['epoch_label_accs']['val']
                    epoch_acum_accs = statistics['epoch_acum_accs']
                    epoch_loss = statistics['epoch_loss']['val']
                    num_epochs = len(epoch_loss)
                    final_acc.append(epoch_acum_accs['val']['acc'][-1])
                    labels = [key for key in epoch_label_accs][:-2]
                    for label in labels:
                        val_acc = epoch_label_accs[label]
                        li = []
                        for epoch in range(num_epochs):
                            acc = val_acc['acc'][epoch]
                            li.append([model_name, imcount, iteration, acc, epoch, scene, label])
                        _df = pd.DataFrame(li, columns=['Methods', 'number of images', 'cv iteration', 'Validation acc',
                                                        'epoch', 'scene', 'label'])
                        data = pd.concat([data, _df], ignore_index=True)

                        li = []
                        for epoch in range(num_epochs):
                            acc = epoch_acum_accs['val']['acc'][epoch]
                            acc_train = epoch_acum_accs['train']['acc'][epoch]
                            li.append([model_name, imcount, iteration, acc, acc_train, epoch, scene])
                        _df = pd.DataFrame(li, columns=['Methods', 'number of images', 'cv iteration', 'Validation acc',
                                                        'Train acc', 'epoch', 'scene'])
                        data_acum_acc = pd.concat([data_acum_acc, _df], ignore_index=True)
                final_acc = np.array(final_acc) * 100
                mean = sum(final_acc) / len(final_acc)
                variance = sum((xi - mean) ** 2 for xi in final_acc) / len(final_acc)
                std = np.sqrt(variance)
                li = [model_name, imcount, scene, mean, variance, std]
                _df = pd.DataFrame([li], columns=['Methods', 'number of images', 'scene', 'mean', 'variance', 'std'])
                data_ev = pd.concat([data_ev, _df], ignore_index=True)

    # print(tabulate(data_ev, headers='keys', tablefmt='psql'))
    os.makedirs(out_path, exist_ok=True)
    data.to_csv(out_path + 'label_acc_over_epoch.csv')
    data_acum_acc.to_csv(out_path + 'mean_acc_over_epoch.csv')
    data_ev.to_csv(out_path + 'mean_variance_comparison.csv')
    data_transfer_classification.to_csv(out_path + 'transfer_classification.csv')
    data_transfer_classification_ev.to_csv(out_path + 'mean_variance_transfer_classification.csv')

    # plot over count
    # plot_acc_over_count(data, out_path)

    # over epoch
    # plot_label_acc_over_epochs(data, out_path)
    # plot_label_acum_acc_over_epoch(data_acum_acc, out_path)

    # plot table
    # csv_to_tex_table(out_path + 'mean_variance_comparison.csv')

    # transfer classification comparison
    # transfer_train_comparison(out_path)
    # csv_to_tex_table(out_path + 'mean_variance_transfer_classification.csv')


def transfer_train_comparison(out_path):
    with open(out_path + 'transfer_classification.csv', 'r') as f:
        os.makedirs(out_path, exist_ok=True)

        data = pd.read_csv(f)
        data['Validation acc'] = data['Validation acc'].apply(lambda x: x / 100)
        data = data.loc[data['scene'] != 'base_scene']

        im_count = data['number of images'].unique()
        colors = sns.color_palette()
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots()

        for c, count in enumerate(im_count):
            data_tmp = data.loc[data['number of images'] == count]
            # Show each observation with a scatterplot
            sns.stripplot(x='Validation acc', y='Methods', hue='scene', data=data_tmp,
                          dodge=True,
                          alpha=.25,
                          zorder=1,
                          jitter=False,
                          palette=[colors[c]]
                          )

        # Show the conditional means, aligning each pointplot in the
        # center of the strips by adjusting the width allotted to each
        # category (.8 by default) by the number of hue levels
        sns.pointplot(x='Validation acc', y='Methods', hue='number of images', data=data,
                      dodge=False,
                      join=False,
                      # palette="dark",
                      markers="d",
                      scale=.75,
                      ci=None
                      )

        with open(out_path + 'label_acc_over_epoch.csv', 'r') as f:
            data_t = pd.read_csv(f)
            data_t = data_t.loc[data_t['scene'] != 'base_scene']
            data_t = data_t.loc[data_t['epoch'] == 24]

            # Show the conditional means, aligning each pointplot in the
            # center of the strips by adjusting the width allotted to each
            # category (.8 by default) by the number of hue levels
            sns.pointplot(x='Validation acc', y='Methods', hue='number of images', data=data_t,
                          dodge=False,
                          join=False,
                          # palette="dark",
                          markers="X",
                          scale=.75,
                          ci=None
                          )

        # plt.title('Comparison of Supervised learning methods')
        # Improve the legend
        handles, labels = ax.get_legend_handles_labels()
        length = len(handles) // 2
        # length = 0

        ax.legend(handles[-3:], labels[-3:], title="number of images",
                  handletextpad=0,
                  loc="upper left", frameon=True)

        ax.set_ylabel("Subsymbolic methods")
        ax.set_xlabel("Classification accuracy")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Resnet-18', 'EfficientNet', 'Vision Transformer'])
        # ax.set_xlim([0.5, 1])
        out_path += 'transfer_classification/'
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(out_path + f'transfer_classification_comparison.png', bbox_inches='tight', dpi=400)
        plt.close()


def plot_acc_over_count(data, out_path, x='Validation acc', y='Methods', hue='scene'):
    with open(out_path + 'label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)

        data_t = data.loc[data['epoch'] == 24]
        scenes = data['scene'].unique()
        im_count = data['number of images'].unique()
        colors = sns.color_palette()
        sns.set_theme(style="whitegrid")
        f, ax = plt.subplots()
        for c, count in enumerate(im_count):
            data_tmp = data_t.loc[data_t['number of images'] == count]

            # Show each observation with a scatterplot
            sns.stripplot(x=x, y=y, hue=hue, data=data_tmp,
                          dodge=True,
                          alpha=.25,
                          zorder=1,
                          jitter=False,
                          palette=[colors[c]]
                          )

        # Show the conditional means, aligning each pointplot in the
        # center of the strips by adjusting the width allotted to each
        # category (.8 by default) by the number of hue levels
        sns.pointplot(x=x, y=y, hue='number of images', data=data_t,
                      dodge=False,
                      join=False,
                      # palette="dark",
                      markers="d",
                      scale=.75,
                      ci=None
                      )

        # plt.title('Comparison of Supervised learning methods')
        # Improve the legend
        handles, labels = ax.get_legend_handles_labels()
        length = len(handles) // 2
        # length = 0

        ax.legend(handles[-3:], labels[-3:], title="number of images",
                  handletextpad=0,
                  loc="upper left", frameon=True)

        ax.set_ylabel("Subsymbolic methods")
        ax.set_xlabel("Validation accuracy")
        ax.set_yticks([0, 1, 2])
        ax.set_yticklabels(['Resnet-18', 'EfficientNet', 'Vision Transformer'])
        ax.set_xlim([0.5, 1])
        os.makedirs(out_path, exist_ok=True)
        plt.savefig(out_path + f'neural_lr_mean_variance.png', bbox_inches='tight', dpi=400)

        plt.close()


def plot_compare_all(data, out_path):
    sns.axes_style("whitegrid")
    plt.gcf()
    sns.lineplot(x="epoch", y="Validation acc",
                 hue="scene",
                 style="number of images",
                 data=data)
    # ax.set_ylim([0, 1])
    # plt.title('method trained on michalski train direction')
    plt.savefig(out_path + 'cross_val_model_scene_imcount_comparison.png', bbox_inches='tight', dpi=400)
    plt.close()


def plot_label_acc_over_epochs(out_path):
    with open(out_path + 'label_acc_over_epoch.csv', 'r') as f:
        data = pd.read_csv(f)
        scenes = data['scene'].unique()
        im_count = data['number of images'].unique()
        raw_trains = 'RandomTrains' if 'RandomTrains' in out_path else 'MichalskiTrains'
        from michalski_trains.dataset import get_datasets
        dataset = get_datasets('base_scene', raw_trains, 10000, y_val='attribute')
        baselines = get_baseline(dataset)
        data.loc[data['label'] == 'load_1', 'label'] = 'load_obj'
        data.loc[data['label'] == 'load_2', 'label'] = 'load_obj'
        data.loc[data['label'] == 'load_3', 'label'] = 'load_obj'
        labels = data['label'].unique()
        colors = sns.color_palette()
        out_path += 'label_acc_over_epoch/'
        os.makedirs(out_path, exist_ok=True)
        for scene in scenes:
            data_tmp = data.loc[data['scene'] == scene]
            for count in im_count:
                data_t = data_tmp.loc[data_tmp['number of images'] == count]
                f, ax = plt.subplots()

                sns.axes_style("whitegrid")
                plt.gcf()
                sns.lineplot(x="epoch", y="Validation acc",
                             hue="label",
                             data=data_t)
                for l_c, label in enumerate(labels):
                    plt.plot(np.arange(25), [baselines['acc'][label]] * 25,
                             dash_capstyle='round', color=colors[l_c], linestyle='dashed')
                ax.set_xlim([0, 24])
                ax.set_ylim(top=1)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Validation accuracy")
                handles, labs = ax.get_legend_handles_labels()
                labs.append('baselines')
                handles.append(Line2D([0], [0], color='black', dash_capstyle='round', linestyle='dashed'))
                ax.legend(handles, labs, title="Train attributes",
                          handletextpad=1,
                          loc="lower right", frameon=True)
                plt.savefig(out_path + f'{scene}_{count}_images_label_acc.png', bbox_inches='tight', dpi=400)
                plt.close()


def plot_label_acum_acc_over_epoch(out_path):
    with open(out_path + 'mean_acc_over_epoch.csv', 'r') as f:
        data_acum = pd.read_csv(f)
        methods = data_acum['Methods'].unique()
        out_path += 'acum_acc_over_epoch/'
        os.makedirs(out_path, exist_ok=True)
        for method in methods:
            tmp1 = data_acum.loc[data_acum['Methods'] == method]
            sns.axes_style("whitegrid")
            f, ax = plt.subplots()
            plt.gcf()
            sns.lineplot(x="epoch", y="Validation acc",
                         hue="scene",
                         style="number of images",
                         data=tmp1)
            ax.set_xlim([0, 24])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Validation accuracy")
            plt.legend(loc="lower right")
            plt.savefig(out_path + f'{method}_plot_val_acc_over_epoch.png', bbox_inches='tight', dpi=400)
            plt.close()

            # plot train acc for method
            sns.axes_style("whitegrid")
            f, ax = plt.subplots()
            plt.gcf()
            sns.lineplot(x="epoch", y="Train acc",
                         hue="scene",
                         style="number of images",
                         data=tmp1)
            ax.set_xlim([0, 24])
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Train accuracy")
            plt.legend(loc="lower right")
            plt.savefig(out_path + f'{method}_plot_train_acc_over_epoch.png', bbox_inches='tight', dpi=400)
            plt.close()


def plot_label_imcount(data, out_path):
    scenes = data['scene'].unique()
    im_count = data['number of images'].unique()
    for scene in scenes:
        data_tmp = data.loc[data['scene'] == scene]
        for count in im_count:
            data_t = data_tmp.loc[data_tmp['number of images'] == count]
            f, ax = plt.subplots()

            sns.axes_style("whitegrid")
            plt.gcf()
            sns.lineplot(x="epoch", y="Validation acc",
                         hue="label",
                         data=data_t)
            # ax.set_ylim([0, 1])
            # plt.title('method trained on michalski train direction')
            ax.set_ylabel("Training Epoch")
            ax.set_ylabel("Validation accuracy")
            plt.legend(title='Train attributes')
            plt.savefig(out_path + f'{scene}_{count}_images_label_acc.png', bbox_inches='tight', dpi=400)
            plt.close()


def csv_to_tex_table(path):
    with open(path, 'r') as f:
        df = pd.read_csv(f)
        scenes = df['scene'].unique()
        im_count = df['number of images'].unique()
        im_count.sort()
        methods = df['Methods'].unique()
        print(tabulate(df, headers='keys', tablefmt='psql'))
        print('tex format:')
        print(r'\begin{table}[H]')
        print(r'    \begin{center}')
        print(r'		\begin{tabular}{l|c|c|r}')
        for method in methods:
            tmp1 = df.loc[df['Methods'] == method]
            print(r'		\multicolumn{4}{c}{%s} \\' % method)
            print('		\hline')
            print(r'		& \textbf{%d images} & \textbf{%d images} & \textbf{%d images} \\' % (
                im_count[0], im_count[1], im_count[2]))
            print('		\hline')

            for scene in scenes:
                tmp2 = tmp1.loc[tmp1['scene'] == scene]
                scene_str = scene.replace('_', ' ')
                text_str = r'		\textbf{%s}           ' % scene_str
                for count in im_count:
                    data = tmp2.loc[tmp2['number of images'] == count]
                    mean = data['mean']
                    std = data['std']

                    text_str += '& \SI{%.2f \pm %.2f}{\percent}        ' % (mean, std)
                print(text_str + r'\\')
        print(r'		\end{tabular}')
        print(r'    \label{tab:tab}')
        print(r'    \end{center}')
        print(r'\end{table}')
