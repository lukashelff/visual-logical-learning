import matplotlib.lines as mlines
import matplotlib.patches as mpatches
from matplotlib import pyplot as plt


def make_3_im_legend(fig, axes, category, category_name, models, colors, mt, legend_h_offset=0):
    axes[1, 1].set_axis_off()
    color_markers = [mlines.Line2D([], [], color=colors[c], marker='d', linestyle='None', markersize=5) for c in
                     category]
    white = [mlines.Line2D([], [], color='white', marker='X', linestyle='None', markersize=0)]
    plt.rcParams.update({'hatch.color': 'black'})

    handels = [mpatches.Patch(facecolor='grey', hatch=mt[m]) for m in models]
    dif = max(0, len(models) - len(category))
    leg = fig.legend(
        white + color_markers + white * (dif + 1) + handels,
        [f'{category_name}:'] + category + [''] * dif + ['Models:'] + [m.title() for m in models],
        loc='lower left',
        bbox_to_anchor=(.515, 0.248+legend_h_offset),
        frameon=True,
        handletextpad=0,
        ncol=2, handleheight=1.2, handlelength=2.5
    )
    for vpack in leg._legend_handle_box.get_children():
        for hpack in vpack.get_children()[:1]:
            hpack.get_children()[0].set_width(0)
