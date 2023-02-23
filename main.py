import argparse
import sys
from itertools import product

import torch


def parse():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='The Michalski Train Problem')
    parser.add_argument('--dataset_size', type=int, default=12000, help='Size of the dataset')
    parser.add_argument('--ds_path', type=str, default="TrainGenerator/output/image_generator",
                        help='path to the dataset directories')
    parser.add_argument('--classification_rule', type=str, default='theoryx',
                        help='the classification rule of the dataset, possible options: '
                             '\'theoryx\', \'easy\', \'color\', \'numerical\', \'multi\', \'complex\', \'custom\'')
    parser.add_argument('--description', type=str, default='MichalskiTrains',
                        help='type of descriptions either \'MichalskiTrains\' or \'RandomTrains\'')
    parser.add_argument('--visualization', type=str, default='Trains', help='Visualization typ of the dataset: '
                                                                            '\'Trains\' or \'SimpleObjects\'')
    parser.add_argument('--background_scene', type=str, default='base_scene',
                        help='dataset Scene: base_scene, desert_scene, sky_scene or fisheye_scene')
    parser.add_argument('--model_name', type=str, default='resnet18',
                        help='model to use for training: \'tesnet18\', \'VisionTransformer\' or \'EfficientNet\'')

    parser.add_argument('--cuda', type=int, default=0, help='Which cuda device to use or cpu if -1')
    parser.add_argument('--command', type=str, default='cnn',
                        help='command ot execute: \'compare_models\' \'cnn\', \'vis\', \'ilp\' or \'ct\'')

    args = parser.parse_args()

    return args


def main():
    args = parse()

    device = torch.device("cpu" if not torch.cuda.is_available() or args.cuda == -1 else f"cuda:{args.cuda}")

    # michalski train dataset settings
    raw_trains = args.description
    train_vis = args.visualization
    base_scene = args.background_scene
    ds_path = args.ds_path
    class_rule = args.classification_rule
    ds_size = args.dataset_size
    model_name = args.model_name
    command = args.command

    sys.path.insert(0, 'TrainGenerator/')
    sys.path.insert(0, 'ilp/rdm-master/')

    if command == 'vis':
        from ds_helper.michalski_3d import get_datasets
        full_ds = get_datasets(base_scene, raw_trains, train_vis, 10, class_rule=class_rule, ds_path=ds_path)
        from visualization.vis_image import show_masked_im

        show_masked_im(full_ds)

    if command == 'ct':
        from raw.concept_tester import eval_rule
        eval_rule()

    if command == 'ilp_crossval':
        from ilp.trainer import Ilp_trainer
        trainer = Ilp_trainer()
        rules = ['theoryx', 'numerical', 'complex']
        models = [args.model_name] if args.model_name == 'popper' or args.model_name == 'aleph' else ['popper', 'aleph']
        train_count = [100, 1000, 10000]
        noise = [0, 0.1, 0.3]
        trainer.cross_val(raw_trains, folds=5, rules=rules, models=models, train_count=train_count, noise=noise,
                          log=False, complete_run=True)

    if command == 'ilp':
        from ilp.trainer import Ilp_trainer
        trainer = Ilp_trainer()
        train_size, val_size = 1000, 2000
        model = 'popper'
        class_rule = 'theoryx'
        trainer.train(model, raw_trains, class_rule, train_size, val_size, noise=0.3, train_log=True)

    if command == "ilp_plot":
        # from ilp.trainer import Ilp_trainer
        # trainer = Ilp_trainer()
        # trainer.plot_ilp_crossval()
        ilp_pth = 'output/ilp'
        ilp_att_noise_pth = 'output/ilp_att_noise'
        from ilp.visualization.noise import plot_noise_robustness
        plot_noise_robustness(ilp_pth)
        from ilp.visualization.vis_bar import plot_ilp_bar
        plot_ilp_bar(ilp_pth)

    if command == 'split_ds':
        from ilp.setup import setup_alpha_ilp_ds
        for rule in ['theoryx', 'numerical', 'complex']:
            setup_alpha_ilp_ds(base_scene, raw_trains, train_vis, ds_size, ds_path, rule)

    if command == 'cnn':
        from models.trainer import Trainer
        resize = False
        batch_size = 25
        lr = 0.001
        rules = ['theoryx', 'numerical', 'complex']
        train_size = [100, 1000, 10000]
        noises = [0, 0.1, 0.3]
        visualizations = ['Trains', 'SimpleObjects']
        scenes = ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']
        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001

        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize, lr=lr)

        trainer.cross_val_train(train_size=train_size, noises=noises, rules=rules, replace=False, save_models=True)
        # trainer.plt_cross_val_performance(True, models=['resnet18', 'EfficientNet', 'VisionTransformer'])

    if command == 'cnn_generalization':
        from models.trainer import Trainer
        resize = False
        batch_size = 25
        lr = 0.001
        rules = ['theoryx', 'numerical', 'complex']
        train_size = [100, 1000, 10000]
        noises = [0, 0.1, 0.3]
        visualizations = ['Trains', 'SimpleObjects']
        scenes = ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']
        min_car = 7
        min_car, max_car = 2, 4

        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001

        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=12000,
                          setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize, lr=lr, resume=True,
                          min_car=min_car, max_car=max_car)
        pth = trainer.update_model_path(prefix=True, im_count=10000, suffix=f'it_{0}/')
        trainer.train()


    if command == 'cnn_plot':
        out_path = 'output/model_comparison/'
        class_rules = ['numerical', 'theoryx', 'complex']
        visuals = ['SimpleObjects', 'Trains']
        from visualization.vis_model_comparison import rule_comparison
        from visualization.vis_bar import plot_sinlge_box, plot_multi_box
        from visualization.data_handler import get_cv_data
        # get_cv_data(f'{out_path}/', 'direction')
        # for rule, vis in product(class_rules, visuals):
        #     plot_sinlge_box(rule, vis, out_path)
        # for rule in class_rules:
        #     plot_multi_box(rule, visuals, out_path)
        # from visualization.vis_point import plot_neural_noise
        # plot_neural_noise(out_path)
        # rule_comparison(out_path)
        from visualization.vis_bar import plot_rules_bar
        plot_rules_bar(out_path, vis='Trains')

        from visualization.ilp_and_neural import vis_ilp_and_neural
        ilp_pth = 'output/ilp'

        vis_ilp_and_neural(out_path, ilp_pth)



    if command == 'perception':
        from models.trainer import Trainer
        model_name = 'resnet18'
        batch_size = 1
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val='attribute', resume=False, batch_size=batch_size)
        trainer.train()


if __name__ == '__main__':
    main()
