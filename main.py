import argparse
import sys
import torch


def parse():
    # Instantiate the parser
    parser = argparse.ArgumentParser(description='The Michalski Train Problem')
    parser.add_argument('--dataset_size', type=int, default=12000, help='Size of the dataset')
    parser.add_argument('--ds_path', type=str, default="TrainGenerator/output/image_generator",
                        help='path to the dataset directories')

    # dataset settings
    parser.add_argument('--rule', type=str, default='theoryx',
                        help='the classification rule of the dataset, possible options: '
                             '\'theoryx\', \'easy\', \'color\', \'numerical\', \'multi\', \'complex\', \'custom\'')
    parser.add_argument('--description', type=str, default='MichalskiTrains',
                        help='type of descriptions either \'MichalskiTrains\' or \'RandomTrains\'')
    parser.add_argument('--visualization', type=str, default='Trains', help='Visualization typ of the dataset: '
                                                                            '\'Trains\' or \'SimpleObjects\'')
    parser.add_argument('--background', type=str, default='base_scene',
                        help='dataset Scene: base_scene, desert_scene, sky_scene or fisheye_scene')
    parser.add_argument('--max_train_length', type=int, default=4, help='max number of cars a train can have')
    parser.add_argument('--min_train_length', type=int, default=2, help='min number of cars a train can have')

    # model settings
    parser.add_argument('--model', type=str, default='resnet18',
                        help='model to use for training: \'resnet18\', \'VisionTransformer\' or \'EfficientNet\'')

    parser.add_argument('--cuda', type=int, default=0, help='Which cuda device to use or cpu if -1')
    parser.add_argument('--action', type=str, default='train',
                        help='command ot execute: \'plot\',\'train\'')
    parser.add_argument('--command', type=str, default='train',
                        help='specific command to execute: \'train\', \'eval\', \'ilp\', \'ilp_crossval\', \'split_ds\','
                             ' \'eval_generalization\' or \'ct\'')
    parser.add_argument('--y_val', type=str, default='direction',
                        help='y value to predict: \'direction\', \'mask\', \'attribute\'')

    args = parser.parse_args()

    return args


def main():
    args = parse()

    device = torch.device("cpu" if not torch.cuda.is_available() or args.cuda == -1 else f"cuda:{args.cuda}")

    # michalski train dataset settings
    raw_trains = args.description
    train_vis = args.visualization
    base_scene = args.background
    ds_path = args.ds_path
    class_rule = args.rule
    ds_size = args.dataset_size
    model_name = args.model
    command = args.command
    action = args.action
    y_val = args.y_val

    sys.path.insert(0, 'TrainGenerator/')
    sys.path.insert(0, 'ilp/rdm-master/')

    if action == 'plot':
        from plotter import plot
        plot(args)
        return

    if action == 'train':
        from train_functions import train
        train(args)
        return

    if action == 'ct':
        from raw.concept_tester import eval_rule
        eval_rule()

    if action == 'eval_generalization':
        min_cars, max_cars = 7, 7
        # min_car, max_car = 2, 4
        ds_size = 2000
        train_vis = 'Trains'
        from models.evaluation import ilp_generalization_test, generalization_test
        ilp_pt = 'output/ilp'
        neural_path = 'output/model_comparison'
        # get generalization results for neural networks
        generalization_test(min_cars, max_cars, base_scene, raw_trains, train_vis, device, ds_path, ds_size=None)
        # get generalization results for ilp
        ilp_generalization_test(ilp_pt, min_cars, max_cars)

    if action == 'split_ds':
        from ilp.setup import setup_alpha_ilp_ds
        for rule in ['theoryx', 'numerical', 'complex']:
            setup_alpha_ilp_ds(base_scene, raw_trains, train_vis, ds_size, ds_path, rule)


if __name__ == '__main__':
    main()
