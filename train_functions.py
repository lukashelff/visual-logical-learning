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
    noise = float(args.noise) / 100
    device = torch.device("cpu" if not torch.cuda.is_available() or args.cuda == -1 else f"cuda:{args.cuda}")
    device = torch.device("cpu" if not torch.cuda.is_available() or args.cuda == -1 else f"cuda")
    y_val = args.y_val
    tag = args.tag

    if command == 'ilp_crossval':
        from ilp.trainer import Ilp_trainer
        trainer = Ilp_trainer()
        rules = ['theoryx', 'numerical', 'complex']
        models = [model_name] if model_name == 'popper' or model_name == 'aleph' else ['popper', 'aleph']
        train_count = [100, 1000, 10000]
        # train_count = [1000]
        noise = [0, 0.1, 0.3]

        trainer.cross_val(raw_trains, folds=5, rules=rules, models=models, train_count=train_count, noise=noise,
                          log=False, complete_run=True)

    if command == 'ns_crossval':
        from models.neuro_symbolic.ns_pipe import inference
        rules = ['theoryx', 'numerical', 'complex']
        sample_sizes = [100, 1000, 10000][1:2]
        noise = [0, 0.1, 0.3][1:]
        inference(train_vis, device, ds_path, ds_size, rules, min_cars, max_cars, sample_sizes=sample_sizes,
                  noise=noise)

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
        rules = [class_rule]
        train_size = [100, 1000, 10000][2:]
        # noises = [0, 0.1, 0.3]
        noises = [noise]
        visualizations = ['Trains', 'SimpleObjects']
        scenes = ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']
        for model in ['resnet18', 'EfficientNet', 'VisionTransformer']:
            if model == 'EfficientNet':
                batch_size = 25
            elif model == 'VisionTransformer':
                resize = True
                lr = 0.00001
            trainer = Trainer(base_scene, raw_trains, train_vis, device, model, class_rule, ds_path, ds_size=ds_size,
                              setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize, lr=lr)
            trainer.cross_val_train(train_size=train_size, label_noise=noises, rules=rules, replace=True,
                                    save_models=False)
        # trainer.plt_cross_val_performance(True, models=['resnet18', 'EfficientNet', 'VisionTransformer'])

    if command == 'image_noise':
        from models.trainer import Trainer
        start_it = 0
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

    if command == 'rcnn_train':
        from models.trainer import Trainer
        batch_size = 10 if torch.cuda.get_device_properties(0).total_memory > 9000000000 else 1
        num_epochs = 30
        train_size, val_size = 10000, 2000
        # every n training steps, the learning rate is reduced by gamma
        num_batches = (train_size * num_epochs) // batch_size
        step_size = num_batches // 3
        lr = 0.001
        gamma = 0.1
        v2 = 'v2'
        model_name = ['rcnn', 'multi_head_rcnn', 'multi_label_rcnn'][2]
        y_val = f'mask{v2}'
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path,
                          ds_size=ds_size, train_size=train_size, val_size=val_size, model_tag=tag,
                          y_val=y_val, resume=False, batch_size=batch_size, setup_model=False, setup_ds=False,
                          num_epochs=num_epochs, gamma=gamma, lr=lr, step_size=step_size, optimizer_='ADAMW')
        model_path = f"output/models/multi_label_rcnn/mask{v2}_classification/{train_vis}_theoryx_RandomTrains_base_scene/imcount_12000_X_val_image_pretrained_lr_0.001_step_10000_gamma0.1/"
        # trainer.setup_ds(val_size=ds_size)
        # trainer.setup_model(resume=True, path=model_path)
        # from models.rcnn.inference import infer_symbolic
        # infer_symbolic(trainer.model, trainer.dl['val'], device, segmentation_similarity_threshold=.8, samples=10,
        #                debug=True)

        # train_size, val_size = 1000, 200
        trainer.train(set_up=True, train_size=train_size, val_size=val_size, ex_name=f'{model_name}_train')

    if command == 'rcnn_train_parallel':
        from models.trainer import Trainer
        batch_size = 5
        num_epochs = 20
        model_name = ['rcnn', 'multi_head_rcnn', 'multi_label_rcnn'][2]
        train_size, val_size = 10000, 2000
        num_batches = (train_size * num_epochs) // batch_size
        lr = 0.01
        # every n training steps, the learning rate is reduced by gamma
        step_size = num_batches // 4
        gamma = 0.1
        gpu_count = 3

        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=False, batch_size=batch_size, setup_model=False, setup_ds=False,
                          num_epochs=num_epochs, gamma=gamma, lr=lr, step_size=step_size, optimizer_='ADAMW')
        trainer.train(set_up=True, train_size=train_size, val_size=val_size, ex_name=f'mul_head_mask_{command}',
                      gpu_count=gpu_count)

    if command == 'rcnn_test':
        from models.trainer import Trainer
        batch_size = 20
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          y_val=y_val, resume=True, batch_size=batch_size, setup_model=True, setup_ds=True)
        from models.rcnn.plot_prediction import predict_and_plot
        predict_and_plot(trainer.model, trainer.dl['val'], device)

    if command == 'rcnn_infer_debug':
        from models.trainer import Trainer
        batch_size = 5
        num_epochs = 20
        train_size, val_size = 12000, 2000
        # every n training steps, the learning rate is reduced by gamma
        num_batches = (train_size * num_epochs) // batch_size
        step_size = num_batches // 3
        lr = 0.001
        gamma = 0.1
        model_name = ['rcnn', 'multi_head_rcnn', 'multi_label_rcnn'][2]
        samples = 100
        v2 = 'v2'
        y_val = f'mask{v2}'
        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path, ds_size=ds_size,
                          lr=lr, step_size=step_size, gamma=gamma, min_car=min_cars, max_car=max_cars,
                          y_val=y_val, resume=True, batch_size=batch_size, setup_model=False, setup_ds=False)
        model_path = f"output/models/multi_label_rcnn/mask{v2}_classification/{train_vis}_theoryx_RandomTrains_base_scene/imcount_12000_X_val_image_pretrained_lr_0.001_step_10000_gamma0.1/"
        trainer.setup_ds(val_size=ds_size)
        trainer.setup_model(resume=True, path=model_path)
        from models.rcnn.inference import infer_symbolic
        infer_symbolic(trainer.model, trainer.dl['val'], device, segmentation_similarity_threshold=.8, samples=samples,
                       debug=True)

    if command == 'train_dtron':
        from detectron2.modeling import build_model
        from models.rcnn.detectron import setup, register_ds, do_train
        from michalski_trains.dataset import get_datasets
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

    if command == 'track_system_stats':
        from ilp.trainer import Ilp_trainer
        # from memory_profiler import memory_usage
        import os
        trainer = Ilp_trainer()
        # noise = [0]
        folds = 1
        rules = ['custom']
        train_count = [100, 1000, 10000]
        model = 'aleph'
        ds_path = f'output/neuro-symbolic/datasets'
        sys_stats_path = f'output/neuro-symbolic/system_stats'
        os.makedirs(sys_stats_path, exist_ok=True)
        print_stats = False
        noise = 0
        visualization = ['Trains', 'SimpleObjects'][0]
        for t_c in train_count:
            for r in rules:
                for f in range(folds):
                    interval, timeout = 5, 60 * 60 * 24 * 7
                    p = f'{ds_path}/{r}/{visualization}_{raw_trains}{t_c}_{noise}noise/cv_{f}'
                    # mem_usage = memory_usage(-1, interval=5, timeout=60*60*24*7)
                    print(f'aleph training on {p}')
                    theory, stats = trainer.aleph_train(path=p, print_stats=print_stats)
                    # memory_usage((trainer.aleph_train, p, {'print_stats': print_stats}),
                    #              interval=interval, timeout=timeout)

                    TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = stats
                    sys_stats = {
                        # 'memory_usage': mem_usage,
                        'theory': theory,
                        'stats': {
                            'TP': TP,
                            'FN': FN,
                            'TN': TN,
                            'FP': FP,
                            'TP_train': TP_train,
                            'FN_train': FN_train,
                            'TN_train': TN_train,
                            'FP_train': FP_train
                        }
                        # 'time': interval * len(mem_usage)
                    }
                    out_p = f'{sys_stats_path}/{visualization}_aleph_{r}_{t_c}smpl_{noise}noise_{f}.json'
                    # save as json
                    import json
                    with open(out_p, 'w+') as f:
                        json.dump(sys_stats, f, indent=4)
