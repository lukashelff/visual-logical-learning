from itertools import chain

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
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
def make_1_im_legend(fig, ax, category, category_name, models, colors, mt, fontsize=20, legend_h_offset=0):
    dif = max(0, len(models) - len(category))
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=15) for c in
                     category] + white * (dif + 1)
    plt.rcParams.update({'hatch.color': 'black'})
    fontsize -= 3
    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    ncols = 5
    models = models.tolist()
    t_r1 = ['Models:'] + models[:ncols - 1]
    t_r2 = [''] + models[ncols - 1:]
    t_r2 += [''] * (ncols - len(t_r2))
    t_r3 = [f'{category_name}:'] + category[:ncols - 1]
    t_r3 += [''] * (ncols - len(t_r3))
    s_r1 = white + handels[:ncols - 1]
    s_r2 = white + handels[ncols - 1:]
    s_r2 += white * (ncols - len(s_r2))
    s_r3 = white + color_markers[:ncols - 1]
    s_r3 += white * (ncols - len(s_r3))
    txt, sym = [], []
    for i in range(ncols):
        sym.append(s_r1[i])
        sym.append(s_r2[i])
        sym.append(s_r3[i])
        txt.append(t_r1[i])
        txt.append(t_r2[i])
        txt.append(t_r3[i])
    leg = fig.legend(
        sym, txt,
        loc='lower center',
        bbox_to_anchor=(.5, -.35 + legend_h_offset),
        frameon=True,
        handletextpad=0,
        ncols=ncols, handleheight=1.2, handlelength=2.5, fontsize=fontsize,
    )

    for vpack in leg._legend_handle_box.get_children()[:1]:
        for hpack in vpack.get_children():
            hpack.get_children()[0].set_width(0)
