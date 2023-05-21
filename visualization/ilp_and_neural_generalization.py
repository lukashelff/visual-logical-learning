import glob
import os
import warnings
from itertools import product

import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

from visualization.data_handler import get_ilp_neural_data, read_csv_stats
from visualization.vis_util import make_3_im_legend, make_1_im_legend, make_1_line_im


def generalization_plot(outpath, vis='Trains', min_cars=7, max_cars=7, tr_samples=10000):
    use_materials = False
    fig_path = f'{outpath}/model_comparison/generalization'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    ilp_stats_path = f'{outpath}/ilp/stats'
    neuro_sym_path = f'{outpath}/neuro-symbolic/stats'
    alpha_ilp = f'{outpath}/neuro-symbolic/alphailp/stats'

    data_gen_ilp = read_csv_stats(fig_path + f'/ilp_generalization_{min_cars}_{max_cars}.csv',
                                  train_length='7', noise=0, symb=True)
    data_gen_cnn = read_csv_stats(fig_path + f'/cnn_generalization_{min_cars}_{max_cars}.csv',
                                  train_length='7', noise=0, symb=False)
    data_gen_neuro_symbolic = read_csv_stats(fig_path + f'/neuro_symbolic_generalization_{min_cars}_{max_cars}.csv',
                                             train_length='7', noise=0, symb=False)

    # neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    # ilp_stats_path = f'{ilp_pth}/stats'
    # neuro_sym_path = f'{neuro_symbolic_path}/stats'

    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_sym_path, alpha_ilp, vis)

    data = pd.concat([data, data_gen_ilp, data_gen_cnn, data_gen_neuro_symbolic], ignore_index=True)
    # data = pd.concat([data, data_gen_ilp, data_gen_cnn], ignore_index=True)
    data = data.loc[data['training samples'] == tr_samples].loc[data['image noise'] == 0].loc[
        data['label noise'] == 0].loc[data['visualization'] == 'Michalski']

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    train_lengths = list(data['Train length'].unique())
    rules = data['rule'].unique()
    models = neural_models + neuro_symbolic_models + ilp_models

    colors_category = train_lengths if use_materials else models
    colors_category_name = 'Train length' if use_materials else 'Models'

    material_category = models if use_materials else train_lengths
    material_category_name = 'Models' if use_materials else 'Train length'

    make_1_line_im(data, material_category, material_category_name, colors_category, colors_category_name,
                   fig_path + f'/generalization_{tr_samples}_tr_samples.png')



