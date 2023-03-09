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
        noises = [0, 0.1, 0.3]
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
        trainer.cross_val_train(train_size=train_size, image_noise=noises, rules=rules, replace=False,
                                save_models=False)

    if command == 'perception':
        from models.trainer import Trainer
        # model_name = 'resnet18'
        batch_size = 1
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=False, batch_size=batch_size, setup_model=False, setup_ds=False)
        trainer.train(set_up=True, )
