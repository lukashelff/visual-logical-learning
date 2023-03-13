from itertools import product

import torch


def plot(args):
    # michalski train dataset settings
    raw_trains = args.description
    train_vis = args.visualization
    base_scene = args.background
    ds_path = args.ds_path
    class_rule = args.rule
    ds_size = args.dataset_size
    model = args.model
    command = args.command
    max_cars = args.max_train_length
    min_cars = args.min_train_length
    device = torch.device("cpu" if not torch.cuda.is_available() or args.cuda == -1 else f"cuda:{args.cuda}")

    if command == 'label_noise':
        ilp_pt = 'output/ilp'
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        # for s in [10000]:
        for s in [100, 1000, 10000]:
            from visualization.ilp_and_neural_label_noise import label_noise_plot, label_noise_degradation_plot
            label_noise_plot(neural_path, ilp_pt, out_path, training_samples=s)
            label_noise_degradation_plot(neural_path, ilp_pt, out_path, training_samples=s)

    if command == 'image_noise':
        ilp_pt = 'output/ilp'
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        # for s in [10000]:
        y_val = 'direction'
        from visualization.data_handler import get_cv_data
        get_cv_data(neural_path, y_val)
        for s in [100, 1000, 10000]:
            from visualization.ilp_and_neural_image_noise import image_noise_plot
            image_noise_plot(neural_path, ilp_pt, out_path, training_samples=s)

    if command == 'attribute_noise':
        from visualization.ilp_attr_noise import attribute_noise_plot
        ilp_pt = 'output/ilp'
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        for s in [100, 1000, 10000][:1]:
            attribute_noise_plot(neural_path, ilp_pt, out_path, training_samples=s)

    if command == 'zoom':
        ds_p = ds_path + '/zoom7'
        from models.eval import zoom_test
        # zoom_test(min_cars, max_cars, base_scene, raw_trains, train_vis, device, ds_p, ds_size=2000)
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        for s in [100, 1000, 10000]:
            from visualization.neural_zoom import zoom_plot
            zoom_plot(neural_path, out_path, tr_samples=s)

    if command == 'elementary_vs_realistic':
        ilp_pt = 'output/ilp'
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        for s in [100, 1000, 10000]:
            from visualization.ilp_and_neural_elementary_vs_realistic import elementary_vs_realistic_plot
            elementary_vs_realistic_plot(neural_path, ilp_pt, out_path, tr_samples=s)

    if command == 'generalization':
        train_vis = 'Trains'
        from visualization.ilp_and_neural_generalization import generalization_plot
        ilp_pt = 'output/ilp'
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        for s in [100, 1000, 10000]:
            generalization_plot(neural_path, ilp_pt, out_path, tr_samples=s)

    if command == 'rule_complexity':
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        from visualization.ilp_and_neural_rule_complexity import rule_complexity_plot
        ilp_pth = 'output/ilp'
        rule_complexity_plot(out_path, ilp_pth, out_path, im_count=1000)

    if command == 'data_efficiency':
        neural_path = 'output/neural'
        out_path = 'output/model_comparison'
        ilp_pth = 'output/ilp'
        from visualization.ilp_and_neural_data_efficiency import data_efficiency_plot
        data_efficiency_plot(neural_path, ilp_pth, out_path)

    ##############################
    # old code
    if command == 'single_rule':
        class_rules = ['numerical', 'theoryx', 'complex']
        visuals = ['SimpleObjects', 'Trains']
        out_path = 'output/model_comparison/'

        from visualization.vis_model_comparison import rule_comparison
        from visualization.vis_bar import plot_sinlge_box, plot_multi_box
        from visualization.data_handler import get_cv_data
        # get_cv_data(f'{out_path}/', 'direction')
        for rule, vis in product(class_rules, visuals):
            plot_sinlge_box(rule, vis, out_path)
        for rule in class_rules:
            plot_multi_box(rule, visuals, out_path)
        from visualization.vis_point import plot_neural_noise
        plot_neural_noise(out_path)
        rule_comparison(out_path)
        from visualization.vis_bar import plot_rules_bar
        plot_rules_bar(out_path, vis='Trains')

    if command == "ilp_plot":
        # from ilp.trainer import Ilp_trainer
        # trainer = Ilp_trainer()
        # trainer.plot_ilp_crossval()
        ilp_pth = 'output/ilp'
        ilp_att_noise_pth = 'output/ilp_att_noise'
        from ilp.visualization.noise import plot_noise_robustness
        plot_noise_robustness(ilp_pth)
        # from ilp.visualization.vis_bar import plot_ilp_bar
        # plot_ilp_bar(ilp_pth)

    if command == 'plot_mask':
        from michalski_trains.dataset import get_datasets
        full_ds = get_datasets(base_scene, raw_trains, train_vis, ds_size=10, class_rule=class_rule, ds_path=ds_path)
        from visualization.vis_image import show_masked_im
        show_masked_im(full_ds)
