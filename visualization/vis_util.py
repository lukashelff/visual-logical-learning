import math
import os
import warnings
from itertools import chain, product

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns


def make_1_line_im(data, material_category, material_category_name, colors_category, colors_category_name, fig_path,
                   figsize=(20, 2), rules=None):
    labelsize, fontsize = 15, 20
    materials_s = ["//", '\\\\', 'x', "///", '/', '\\', '.', 'o', '+', 'O', '*']
    mt = {model: materials_s[n] for n, model in enumerate(material_category)}

    colors_s = sns.color_palette('dark')
    colors = {vis: colors_s[n] for n, vis in enumerate(colors_category)}
    rules = ['theoryx', 'numerical', 'complex'] if rules is None else rules

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(1, len(rules), wspace=.05, hspace=.15)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]

    for col, rule in enumerate(rules):
        ax = axes[col]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        for c, m in product(colors_category, material_category):
            data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
            try:
                sns.barplot(x=colors_category_name, order=colors_category, y='Validation acc',
                            hue=material_category_name, hue_order=material_category, data=data_temp,
                            palette={m: col for m, col in zip(material_category, ['gray' for _ in material_category])},
                            alpha=.7, ax=ax, orient='v', hatch=mt[m])
            except:
                warnings.warn(f'No data for {c}, {m}, {rule}')
        for bar_group, desaturate_value in zip(ax.containers, [1] * len(ax.containers)):
            ax.bar_label(bar_group, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=3)
            for c_bar, bar in enumerate(bar_group):
                color = colors_s[c_bar]
                # bar.set_color(sns.desaturate(color, desaturate_value))
                bar.set_facecolor(sns.desaturate(color, desaturate_value))
                bar.set_edgecolor('black')
        ax.get_legend().remove()
        ax.set_ylim([50, 111])
        ax.get_xaxis().set_visible(False)
        if col != 0:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=labelsize)

    make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
                     labelsize, legend_h_offset=-0.18, legend_v_offset=0.0)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=400)

    plt.close()


# vertical legend
def make_3_im_legend(fig, axes, category, category_name, models, colors, mt, legend_h_offset=0):
    axes[1, 1].set_axis_off()
    dif = max(0, len(models) - len(category))
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     category] + white * (dif + 1)
    plt.rcParams.update({'hatch.color': 'black'})
    category += [''] * dif
    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    leg = fig.legend(
        white + color_markers + handels,
        [f'{category_name}:'] + category + ['Models:'] + [m.title() for m in models],
        loc='lower left',
        bbox_to_anchor=(.515, 0.248 + legend_h_offset),
        frameon=True,
        handletextpad=.5,
        ncol=2, handleheight=1.2, handlelength=2.5
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)


def make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, materials,
                     material_category_name='Models',
                     fontsize=15, legend_h_offset=0, legend_v_offset=0, ncols=5):
    dif = max(0, len(material_category) - len(colors_category))
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=10) for c in
                     colors_category]
    plt.rcParams.update({'hatch.color': 'black'})
    mt_markers = [mpatches.Patch(facecolor='white', hatch=materials[m], edgecolor='black') for m in material_category]

    color_rows = math.ceil((len(colors_category)) / (ncols - 1))
    mt_rows = math.ceil((len(material_category)) / (ncols - 1))
    nrows = color_rows + mt_rows
    if (color_rows * (ncols - 1) - len(colors_category)) > 0:
        colors_category += [''] * (color_rows * (ncols - 1) - len(colors_category))
        color_markers += white * (color_rows * (ncols - 1) - len(color_markers))
    if (mt_rows * (ncols - 1) - len(material_category)) > 0:
        material_category += [''] * (mt_rows * (ncols - 1) - len(material_category))
        mt_markers += white * (mt_rows * (ncols - 1) - len(mt_markers))

    first_col_txt = [f'{colors_category_name.title()}:'] + [''] * (color_rows - 1) + [
        f'{material_category_name.title()}:'] + [''] * (
                            mt_rows - 1)
    first_col_handles = (color_rows + mt_rows) * white

    txt = np.array(colors_category + material_category).reshape(nrows, (ncols - 1))
    txt = np.concatenate((np.array(first_col_txt).reshape(nrows, 1), txt), axis=1)
    handles = np.array(color_markers + mt_markers).reshape(nrows, (ncols - 1))
    handles = np.concatenate((np.array(first_col_handles).reshape(nrows, 1), handles), axis=1)
    handles = handles.T.flatten().tolist()
    txt = txt.T.flatten().tolist()

    leg = fig.legend(
        handles, txt,
        loc='lower center',
        bbox_to_anchor=(.496 + legend_v_offset, -.35 + legend_h_offset),
        frameon=False,
        handletextpad=.5,
        ncols=ncols, handleheight=1.3, handlelength=2.5, fontsize=fontsize,
    )

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)


def make_3_im(data, material_category, material_category_name, colors_category, colors_category_name, fig_path,
              figsize=(20, 4), rules=None, legend_offset=(0.14, 0.0), legend_cols=5, ):
    labelsize, fontsize = 15, 20
    materials_s = ["//", '\\\\', 'x', "///", '/', '\\', '.', 'o', '+', 'O', '*']
    mt = {model: materials_s[n] for n, model in enumerate(material_category)}

    colors_s = sns.color_palette('dark')
    colors = {vis: colors_s[n] for n, vis in enumerate(colors_category)}
    rules = ['theoryx', 'numerical', 'complex'] if rules is None else rules

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.3)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    completed_runs = []
    for col, rule in enumerate(rules):
        ax = axes[col // 2, col % 2]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        run = []
        y_pos = []
        for c, m in product(colors_category, material_category):
            data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
            run += ['*'] if len(data_temp) < 5 else ['']
            y_pos += [40 if np.isnan(data_temp['Validation acc'].mean()) else data_temp['Validation acc'].mean()]
            try:
                sns.barplot(x=colors_category_name, order=colors_category, y='Validation acc',
                            hue=material_category_name, hue_order=material_category, data=data_temp,
                            palette={m: col for m, col in zip(material_category, ['gray' for _ in material_category])},
                            alpha=.7, ax=ax, orient='v', hatch=mt[m])
            except:
                warnings.warn(f'No data for {c}, {m}, {rule}')
        completed_runs += [run]
        for bar_group, desaturate_value in zip(ax.containers, [1] * len(ax.containers)):
            ax.bar_label(bar_group, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=1)
            for c_bar, bar in enumerate(bar_group):
                color = colors_s[c_bar]
                # bar.set_color(sns.desaturate(color, desaturate_value))
                bar.set_facecolor(sns.desaturate(color, desaturate_value))
                bar.set_edgecolor('black')
        ax.get_legend().remove()
        ax.get_xaxis().set_visible(False)
        ax.set_xlabel('')
        x_pos = sorted(list(set([p.get_x() + p.get_width() / 2 for p in ax.patches])))
        if col % 2:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Accuracy', fontsize=labelsize)
            ax.set_ylim([50, 111])
        for x, y, r in zip(x_pos, y_pos, run):
            ax.text(x, y + 10, r, fontsize=labelsize, ha='center')
    # axes[1, 1].set_axis_off()
    leg = axes[1, 1]
    leg.get_xaxis().set_visible(False)
    leg.get_yaxis().set_visible(False)

    make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
                     labelsize, legend_h_offset=legend_offset[0], legend_v_offset=legend_offset[1], ncols=legend_cols)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=400)
    # save completed runs
    with open(fig_path.replace('.png', '_completed_runs.txt'), 'w') as f:
        f.write(f'Completed runs: {completed_runs}')
    print(f'run {fig_path} completed')
    print(f'Completed runs: {completed_runs}')

    plt.close()


def make_3_im_deg(data, material_category, material_category_name, colors_category, colors_category_name, fig_path,
                  figsize=(20, 4), rules=None, legend_offset=(0.14, 0.0), legend_cols=5):
    labelsize, fontsize = 15, 20
    materials_s = ["//", '\\\\', 'x', "///", '/', '\\', '.', 'o', '+', 'O', '*']
    mt = {model: materials_s[n] for n, model in enumerate(material_category)}

    colors_s = sns.color_palette('dark')
    colors = {vis: colors_s[n] for n, vis in enumerate(colors_category)}
    rules = ['theoryx', 'numerical', 'complex'] if rules is None else rules

    sns.set_theme(style="whitegrid")
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 2, wspace=.05, hspace=.3)
    axes = gs.subplots(sharex=True, sharey=True, )
    axes = axes if isinstance(axes, np.ndarray) else [axes]
    completed_runs = []
    for col, rule in enumerate(rules):
        ax = axes[col // 2, col % 2]
        ax.grid(axis='x')
        ax.set_title(rule.title(), fontsize=fontsize)
        ax.tick_params(bottom=False, left=False, labelsize=labelsize)
        for spine in ax.spines.values():
            spine.set_edgecolor('gray')
        data_t = data.loc[data['rule'] == rule]
        run = []
        y_pos = []
        for c, m in product(colors_category, material_category):
            data_temp = data_t.loc[data[colors_category_name] == c].loc[data[material_category_name] == m]
            run += ['*'] if len(data_temp) < 5 else ['']
            y_pos += [-15 if np.isnan(data_temp['Noise Degradation'].mean()) else data_temp['Noise Degradation'].mean()]

            try:
                sns.barplot(x=colors_category_name, order=colors_category, y='Noise Degradation',
                            hue=material_category_name, hue_order=material_category, data=data_temp,
                            palette={m: col for m, col in zip(material_category, ['gray' for _ in material_category])},
                            alpha=.7, ax=ax, orient='v', hatch=mt[m])
            except:
                warnings.warn(f'No data for {c}, {m}, {rule}')
        completed_runs += [run]
        for bar_group, desaturate_value in zip(ax.containers, [1] * len(ax.containers)):
            ax.bar_label(bar_group, fmt='%1.f', label_type='edge', fontsize=labelsize, padding=1)
            for c_bar, bar in enumerate(bar_group):
                color = colors_s[c_bar]
                # bar.set_color(sns.desaturate(color, desaturate_value))
                bar.set_facecolor(sns.desaturate(color, desaturate_value))
                bar.set_edgecolor('black')
        ax.get_legend().remove()
        ax.get_xaxis().set_visible(False)
        ax.set_xlabel('')
        x_pos = sorted(list(set([p.get_x() + p.get_width() / 2 for p in ax.patches])))
        ax.set_ylim([0, 130])
        if col % 2:
            ax.set_ylabel('')
        else:
            ax.set_ylabel('Degradation (%)', fontsize=labelsize)
        for x, y, r in zip(x_pos, y_pos, run):
            ax.text(x, y + 15, r, fontsize=labelsize, ha='center')
    # axes[1, 1].set_axis_off()
    leg = axes[1, 1]
    leg.get_xaxis().set_visible(False)
    leg.get_yaxis().set_visible(False)
    make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, mt, material_category_name,
                     labelsize, legend_h_offset=legend_offset[0], legend_v_offset=legend_offset[1], ncols=legend_cols)
    os.makedirs(os.path.dirname(fig_path), exist_ok=True)
    plt.savefig(fig_path, bbox_inches='tight', dpi=400)
    # save completed runs
    with open(fig_path.replace('.png', '_completed_runs.txt'), 'w') as f:
        f.write(f'Completed runs: {completed_runs}')
    print(f'run {fig_path} completed')
    print(f'Completed runs: {completed_runs}')

    plt.close()
