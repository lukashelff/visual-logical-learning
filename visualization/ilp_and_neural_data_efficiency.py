from visualization.data_handler import get_ilp_neural_data
from visualization.vis_util import make_1_line_im, make_3_im


def data_efficiency_plot(outpath, vis='Trains'):
    use_materials = False
    fig_path = f'{outpath}/model_comparison/data_efficiency'

    # ilp_stats_path = f'{ilp_pth}/stats'
    # neural_stats_path = neural_path + '/label_acc_over_epoch.csv'
    ilp_stats_path = f'{outpath}/ilp/stats'
    neural_stats_path = f'{outpath}/neural/label_acc_over_epoch.csv'
    neuro_sym_path = f'{outpath}/neuro-symbolic/stats'
    alpha_ilp = f'{outpath}/neuro-symbolic/alphailp/stats'
    data, ilp_models, neural_models, neuro_symbolic_models = get_ilp_neural_data(ilp_stats_path, neural_stats_path,
                                                                                 neuro_sym_path, alpha_ilp, vis)
    models = neural_models + neuro_symbolic_models + ilp_models
    data = \
        data.loc[data['image noise'] == 0].loc[data['label noise'] == 0].loc[data['visualization'] == 'Michalski'].loc[
            data['Train length'] == '2-4']

    scenes = data['scene'].unique()
    im_count = sorted(data['training samples'].unique())
    im_count = [int(i) for i in im_count]
    rules = data['rule'].unique()

    colors_category = im_count if use_materials else models
    colors_category_name = 'training samples' if use_materials else 'Models'

    material_category = models if use_materials else im_count
    material_category_name = 'Models' if use_materials else 'training samples'

    # make_1_line_im(data, material_category, material_category_name, colors_category, colors_category_name,
    #                fig_path + f'/data_efficiency.png', (27, 2))
    make_3_im(data, material_category, material_category_name, colors_category, colors_category_name,
              fig_path + f'/data_efficiency.png', (27, 4), legend_offset=(0.43, 0.213), legend_cols=4)
