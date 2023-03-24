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
                                save_models=False)

    if command == 'perception':
        from models.trainer import Trainer
        # model_name = 'resnet18'
        # batch_size = 10
        batch_size = 15
        lr = 0.001
        # every n training steps, the learning rate is reduced by gamma
        step_size = round(50000 / batch_size)
        weight_decay = 0.1
        num_epochs = 20
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=False, batch_size=batch_size, setup_model=False, setup_ds=False,
                          num_epochs=num_epochs, gamma=weight_decay, lr=lr, step_size=step_size, optimizer_='ADAMW')
        trainer.train(set_up=True, train_size=10000, val_size=2000)

    if command == 'perception_test':
        from models.trainer import Trainer
        batch_size = 20
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=True, batch_size=batch_size, setup_model=True, setup_ds=True)
        from models.rcnn.plot_prediction import predict_and_plot
        predict_and_plot(trainer.model, trainer.dl['val'], device)

    if command == 'perception_infer':
        from models.trainer import Trainer
        batch_size = 1
        samples = 10
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=True, batch_size=batch_size, setup_model=True, setup_ds=True)
        from models.rcnn.inference import infer_symbolic
        infer_symbolic(trainer.model, trainer.dl['val'], device, segmentation_similarity_threshold=.8, samples=10)

    if command == 'train_dtron':
        logger = logging.getLogger("detectron2")
        from detectron2.modeling import build_model
        from models.train_loop import do_train
        from models.rcnn.detectron import setup
        from models.rcnn.detectron import register_ds

        print('train detectron 2')
        print('detectron 2 predicts segmentation and the corresponding label')
        image_count = 10000
        register_ds(base_scene, train_col, image_count)
        cfg_path = "models/rcnn/configs/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = setup(cfg_path, base_scene, train_col)

        model = build_model(cfg)
        logger.info("Model:\n{}".format(model))
        experiment_name = f'dtron_{base_scene[:3]}_{train_col[0]}'

        do_train(cfg, model, experiment_name, resume=False)

    if command == 'test_dtron':
        print('test detectron 2')
        from models.rcnn.detectron import setup
        from models.rcnn.detectron import do_test

        cfg_path = "models/rcnn/configs/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = setup(cfg_path, base_scene, train_col)
        do_test(cfg, base_scene, train_col)

    if command == 'eval_dtron':
        from visualization.vis_detectron import detectron_pred_vis_images, plt_metrics
        from models.rcnn.detectron import setup

        print('evaluate performance of detectron 2')
        cfg_path = "models/rcnn/configs/mask_rcnn_R_101_FPN_3x.yaml"
        cfg = setup(cfg_path, base_scene, train_col)
        detectron_pred_vis_images(cfg, base_scene, train_col)
        plt_metrics(base_scene, train_col)
