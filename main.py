import argparse
import os
import sys
import torch


def main():
    args = parse()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")

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
        from TrainGenerator.michalski_trains import m_train_dataset
        full_ds = m_train_dataset.get_datasets(base_scene, raw_trains, train_vis, 10, class_rule=class_rule,
                                               ds_path=ds_path)
        from visualization.vis import show_masked_im
        show_masked_im(full_ds)

    if command == 'ct':
        from raw.concept_tester import eval_rule
        eval_rule()

    if command == 'ilp_crossval':
        from ilp.trainer import Ilp_trainer
        trainer = Ilp_trainer()
        rules = ['theoryx', 'numerical', 'complex']
        models = ['popper', 'aleph'][:1]
        train_count = [100, 1000, 10000]
        noise = [0, 0.1, 0.3][1:]
        trainer.cross_val(raw_trains, folds=5, rules=rules, models=models, train_count=train_count, noise=noise, log=False, complete_run=True)
        # trainer.plot_ilp_crossval()
        # trainer.plot_noise_robustness()

    if command == 'ilp':
        from ilp.trainer import Ilp_trainer
        trainer = Ilp_trainer()
        train_size, val_size = 1000, 2000
        model = 'popper'
        class_rule = 'theoryx'
        trainer.train(model, raw_trains, class_rule, train_size, val_size, noise=0.3, train_log=True)

    if command == 'split_ds':
        from michalski_trains.m_train_dataset import get_datasets
        import shutil
        ds = get_datasets(base_scene, raw_trains, train_vis, ds_size, ds_path=ds_path, class_rule=class_rule)
        path_train_true = 'output/alphailp-images/train/true'
        path_test_true = 'output/alphailp-images/test/true'
        path_val_true = 'output/alphailp-images/val/true'
        path_train_false = 'output/alphailp-images/train/false'
        path_test_false = 'output/alphailp-images/test/false'
        path_val_false = 'output/alphailp-images/val/false'
        for p in [path_train_true, path_test_true, path_val_true, path_train_false, path_test_false, path_val_false]:
            os.makedirs(p, exist_ok=True)
            p += '/image'
        train_t, test_t, val_t, train_f, test_f, val_f = [0] * 6
        c = 0
        while sum([train_t, test_t, val_t, train_f, test_f, val_f]) < 300:
            path = ds.get_image_path(c)
            label = ds.get_label_for_id(c)
            # print(f'iteration: {c}, label: {label}, path: {path}')
            if label == 'east':
                if train_t < 50:
                    shutil.copyfile(path, path_train_true + f'/image{train_t}.png')
                    train_t += 1
                elif test_t < 50:
                    shutil.copyfile(path, path_test_true + f'/image{test_t}.png')
                    test_t += 1
                elif val_t < 50:
                    shutil.copyfile(path, path_val_true + f'/image{val_t}.png')
                    val_t += 1
            elif label == 'west':
                if train_f < 50:
                    shutil.copyfile(path, path_train_false + f'/image{train_f}.png')
                    train_f += 1
                elif test_f < 50:
                    shutil.copyfile(path, path_test_false + f'/image{test_f}.png')
                    test_f += 1
                elif val_f < 50:
                    shutil.copyfile(path, path_val_false + f'/image{val_f}.png')
                    val_f += 1
            c += 1

    if command == 'cnn':
        from models.trainer import Trainer
        resize = False
        batch_size = 25
        lr = 0.001
        rules = ['theoryx', 'numerical', 'complex']
        rules = ['complex']
        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001

        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize, lr=lr)
        # train_size = [10000]
        trainer.cross_val_train(replace=False, save_models=True)

    if command == 'compare_models':
        model_names = ['resnet18', 'VisionTransformer', 'EfficientNet']
        # model_names = ['resnet18', 'EfficientNet']
        out_path = 'output/model_comparison/'
        class_rules = ['numerical', 'theoryx', 'complex']
        from visualization.vis_model_comparison import rule_comparison
        rule_comparison(model_names, out_path)


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
                        help='model to use for training: \'resnet18\', \'VisionTransformer\' or \'EfficientNet\'')

    parser.add_argument('--cuda', type=int, default=0, help='Which cuda device to use')
    parser.add_argument('--command', type=str, default='cnn',
                        help='command ot execute: \'compare_models\' \'cnn\', \'vis\', \'ilp\' or \'ct\'')

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    main()
