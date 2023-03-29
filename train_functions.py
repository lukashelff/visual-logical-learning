import logging

import torch


def train(args):
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
    max_cars = args.max_train_length
    min_cars = args.min_train_length
    device = torch.device("cpu" if not torch.cuda.is_available() or args.cuda == -1 else f"cuda:{args.cuda}")
    y_val = args.y_val

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

    if command == 'cnn':
        from models.trainer import Trainer
        resize = False
        batch_size = 25
        lr = 0.001
        rules = ['theoryx', 'numerical', 'complex']
        train_size = [100, 1000, 10000]
        noises = [0, 0.1, 0.3]
        noises = [0]
        visualizations = ['Trains', 'SimpleObjects']
        scenes = ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']
        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize, lr=lr)
        trainer.cross_val_train(train_size=train_size, label_noise=noises, rules=rules, replace=False, save_models=True)
        # trainer.plt_cross_val_performance(True, models=['resnet18', 'EfficientNet', 'VisionTransformer'])

    if command == 'image_noise':
        from models.trainer import Trainer
        start_it = 55
        resize = False
        batch_size = 25
        lr = 0.001
        rules = ['theoryx', 'numerical', 'complex']
        train_size = [100, 1000, 10000]
        noises = [0.1, 0.3]
        visualizations = ['Trains', 'SimpleObjects']
        scenes = ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']
        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize, lr=lr)
        trainer.cross_val_train(train_size=train_size, image_noise=noises, rules=rules, replace=True,
                                save_models=False, ex_name=f'{action}_{model_name[:4]}_{command}', start_it=start_it)

    if command == 'perception':
        from models.trainer import Trainer
        # model_name = 'resnet18'
        batch_size = 5
        # batch_size = 1
        lr = 0.001
        # every n training steps, the learning rate is reduced by gamma
        step_size = round(50000 / batch_size)
        gamma = 0.1
        num_epochs = 10
        model_name = 'rcnn'
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=False, batch_size=batch_size, setup_model=False, setup_ds=False,
                          num_epochs=num_epochs, gamma=gamma, lr=lr, step_size=step_size, optimizer_='ADAMW')
        trainer.train(set_up=True, train_size=10000, val_size=2000, ex_name=f'{action}_{model_name[:4]}_{command}')

    if command == 'perception_test':
        from models.trainer import Trainer
        batch_size = 20
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=True, batch_size=batch_size, setup_model=True, setup_ds=True)
        from models.rcnn.plot_prediction import predict_and_plot
        predict_and_plot(trainer.model, trainer.dl['val'], device)

    if command == 'perception_infer':
        from models.trainer import Trainer
        batch_size = 2
        samples = 100
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=False, batch_size=batch_size, setup_model=True, setup_ds=True)
        from models.rcnn.inference import infer_symbolic
        infer_symbolic(trainer.model, trainer.dl['val'], device, segmentation_similarity_threshold=.8, samples=samples,
                       debug=False)

    if command == 'train_dtron':
        from detectron2.modeling import build_model
        from models.rcnn.detectron import setup
        from models.rcnn.detectron import register_ds
        from michalski_trains.dataset import get_datasets
        from models.rcnn.detectron import do_train

        logger = logging.getLogger("detectron2")

        y_val = 'mask'
        print('train detectron 2')
        print('detectron 2 predicts segmentation and the corresponding label')
        image_count = 10000
        full_ds = get_datasets(base_scene, raw_trains, train_vis, class_rule,
                               y_val=y_val, ds_size=ds_size, ds_path=ds_path)
        register_ds(full_ds)
        cfg_path = "models/rcnn/configs/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = setup(cfg_path, base_scene, raw_trains, device)

        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))
        experiment_name = f'dtron_{base_scene[:3]}_{raw_trains[0]}'

        do_train(cfg, model, experiment_name, logger, resume=True)

    if command == 'test_dtron':
        print('test detectron 2')
        from models.rcnn.detectron import setup, do_test
        from models.rcnn.detectron import register_ds
        from michalski_trains.dataset import get_datasets
        logger = logging.getLogger("detectron2")
        cfg_path = "models/rcnn/configs/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = setup(cfg_path, base_scene, raw_trains, device)

        # register dataset
        full_ds = get_datasets(base_scene, raw_trains, train_vis, class_rule, y_val=y_val, ds_size=ds_size,
                               ds_path=ds_path)
        register_ds(full_ds, image_count=10)
        # setup detectron
        res = do_test(cfg, logger)
        print(res)

    if command == 'eval_dtron':
        from visualization.vis_detectron import detectron_pred_vis_images, plt_metrics
        from models.rcnn.detectron import setup
        from models.rcnn.detectron import register_ds
        from michalski_trains.dataset import get_datasets
        logger = logging.getLogger("detectron2")
        cfg_path = "models/rcnn/configs/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = setup(cfg_path, base_scene, raw_trains, device)
        # register dataset
        full_ds = get_datasets(base_scene, raw_trains, train_vis, class_rule, y_val=y_val, ds_size=ds_size,
                               ds_path=ds_path)
        register_ds(full_ds, image_count=10)

        print('evaluate performance of detectron 2')
        detectron_pred_vis_images(cfg)
        plt_metrics(cfg.OUTPUT_DIR)

    if command == 'infer_dtron':
        from visualization.vis_detectron import detectron_pred_vis_images, plt_metrics
        from models.rcnn.detectron import setup, detectron_infer_symbolic, register_ds
        from michalski_trains.dataset import get_datasets
        logger = logging.getLogger("detectron2")
        cfg_path = "models/rcnn/configs/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = setup(cfg_path, base_scene, raw_trains, device)
        # register dataset
        full_ds = get_datasets(base_scene, raw_trains, train_vis, class_rule, y_val=y_val, ds_size=ds_size,
                               ds_path=ds_path)
        register_ds(full_ds, image_count=10)

        print('detectron 2 inferring symbolic representations of the trains')
        detectron_infer_symbolic(cfg)
