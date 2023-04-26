import math
from itertools import chain, product

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
from matplotlib import pyplot as plt


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
        handletextpad=0,
        ncol=2, handleheight=1.2, handlelength=2.5
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)


# horizontal legend
# def make_1_im_legend(fig, ax, category, category_name, models, colors, mt, fontsize=15, legend_h_offset=0, ncols=5):

def make_1_im_legend(fig, colors_category, colors, colors_category_name, material_category, materials, material_category_name='Models',
                     fontsize=15, legend_h_offset=0, legend_v_offset=0, ncols=5):
    dif = max(0, len(material_category) - len(colors_category))
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=10) for c in
                     colors_category]
    plt.rcParams.update({'hatch.color': 'black'})
    mt_markers = [mpatches.Patch(facecolor='grey', hatch=materials[m]) for m in material_category]

    color_rows = math.ceil((len(colors_category)) / (ncols - 1))
    mt_rows = math.ceil((len(material_category)) / (ncols - 1))
    nrows = color_rows + mt_rows

    colors_category += [''] * (color_rows * (ncols - 1) - len(colors_category))
    material_category += [''] * (mt_rows * (ncols - 1) - len(material_category))
    color_markers += white * (color_rows * (ncols - 1) - len(color_markers))
    mt_markers += white * (mt_rows * (ncols - 1) - len(mt_markers))

    # t_r1 = [f'{mt_category_name}:'] + mt_category[:ncols - 1]
    # t_r2 = [''] + mt_category[ncols - 1:]
    # t_r2 += [''] * (ncols - len(t_r2))
    # t_r3 = [f'{colors_category_name}:'] + colors_category[:ncols - 1]
    # t_r3 += [''] * (ncols - len(t_r3))
    # s_r1 = white + patches[:ncols - 1]
    # s_r2 = white + patches[ncols - 1:]
    # s_r2 += white * (ncols - len(s_r2))
    # s_r3 = white + color_markers[:ncols - 1]
    # s_r3 += white * (ncols - len(s_r3))
    # txt, handles = [], []
    # nrows = math.ceil((len(colors_category)+1) // ncols) + math.ceil((len(mt_category)+1) // ncols)
    # for i in range(ncols):
    #     handles.append(s_r1[i])
    #     if t_r2 != [''] * ncols:
    #         handles.append(s_r2[i])
    #     handles.append(s_r3[i])
    #     txt.append(t_r1[i])
    #     if t_r2 != [''] * ncols:
    #         txt.append(t_r2[i])
    #     txt.append(t_r3[i])

    first_col_txt = [f'{colors_category_name.title()}:'] + [''] * (color_rows - 1) + [f'{material_category_name.title()}:'] + [''] * (
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
        frameon=True,
        handletextpad=0,
        ncols=ncols, handleheight=1.3, handlelength=2.5, fontsize=fontsize,
    )

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)
