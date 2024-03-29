import os

import torch


def inference(train_vis, device, ds_path, ds_size, class_rules, min_cars, max_cars, sym_models=None, sample_sizes=None, noise=[0,.1,.3], ilp_timeout=60*60*24):
    from models.trainer import Trainer
    from ilp.trainer import Ilp_trainer
    print('NS PIPE')
    base_scene, raw_trains = 'base_scene', 'MichalskiTrains'
    sym_models = ['popper', 'aleph'] if sym_models is None else sym_models
    sample_sizes = [1000] if sample_sizes is None else sample_sizes
    batch_size = 10 if torch.cuda.get_device_properties(0).total_memory > 9000000000 else 1
    num_epochs = 30
    train_size, val_size = 12000, 2000
    # every n training steps, the learning rate is reduced by gamma
    num_batches = (train_size * num_epochs) // batch_size
    step_size = num_batches // 3
    lr = 0.001
    gamma = 0.1
    v2 = 'v2'
    y_val = f'mask{v2}'
    print('neural inference')
    for rule in class_rules:
        p = f'output/models/multi_label_rcnn/inferred_ds/prediction/{rule}/{train_vis}_MichalskiTrains_len_{min_cars}-{max_cars}.txt'
        if os.path.exists(p):
            print(f'Prediction already exists. skipping inference for {p}')
            continue
        print(f'Inference for {p}')
        trainer = Trainer(base_scene, raw_trains, train_vis, device, 'multi_label_rcnn', rule, ds_path, ds_size=ds_size,
                          lr=lr, step_size=step_size, gamma=gamma, min_car=min_cars, max_car=max_cars,
                          y_val=y_val, resume=True, batch_size=batch_size, setup_model=False, setup_ds=False)
        model_path = f"output/models/multi_label_rcnn/mask{v2}_classification/{train_vis}_theoryx_RandomTrains_base_scene/imcount_12000_X_val_image_pretrained_lr_0.001_step_10000_gamma0.1/"
        trainer.setup_ds(val_size=ds_size)
        trainer.setup_model(resume=True, path=model_path)
        out_path = f'output/models/multi_label_rcnn/inferred_ds/'
        from models.rcnn.inference import infer_dataset
        infer_dataset(trainer.model, trainer.dl['val'], device, out_path, train_vis, rule, 'MichalskiTrains',
                      min_cars, max_cars, )

    trainer = Ilp_trainer()
    # noise = [0]
    output_dir = 'output/neuro-symbolic'
    pred_dir = f'output/models/multi_label_rcnn/inferred_ds/prediction'
    tag = train_vis + '_'
    print('ilp cross-validation')
    trainer.cross_val(raw_trains, folds=5, rules=class_rules, models=sym_models, train_count=sample_sizes, noise=noise,
                      log=False, complete_run=True, output_dir=output_dir, symbolic_ds_path=pred_dir, tag=tag,
                      per_run_timeout=ilp_timeout)
