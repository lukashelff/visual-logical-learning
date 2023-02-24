from itertools import product
import pandas as pd
from models.trainer import Trainer


def generalization_test(min_cars, max_cars, base_scene, raw_trains, train_vis, device, ds_path, ds_size=None):
    ds_size = ds_size if ds_size is not None else 2000
    data = pd.DataFrame(
        columns=['Methods', 'number of images', 'rule', 'visualization', 'scene', 'cv iteration', 'label',
                 'Validation acc', "precision", "recall"])
    for model, rule in product(['resnet18', 'EfficientNet', 'VisionTransformer'],
                               ['theoryx', 'numerical', 'complex']):
        model_name = model
        resize = False
        batch_size = 25
        lr = 0.001
        if model_name == 'EfficientNet':
            batch_size = 25
        elif model_name == 'VisionTransformer':
            resize = True
            lr = 0.00001

        trainer = Trainer(base_scene, raw_trains, train_vis, device, model_name, rule, ds_path,
                          ds_size=ds_size, setup_model=False, setup_ds=False, batch_size=batch_size, resize=resize,
                          lr=lr, resume=True, min_car=min_cars, max_car=max_cars)
        for cv in range(5):
            pth = trainer.get_model_path(prefix=True, im_count=10000, suffix=f'it_{cv}/', model_name=model_name)
            acc, precision, recall = trainer.val(model_path=pth)
            frame = [[model_name, 10000, rule, train_vis, base_scene, cv, 'direction', acc, precision, recall]]
            _df = pd.DataFrame(frame, columns=['Methods', 'number of images', 'rule', 'visualization', 'scene',
                                               'cv iteration', 'label', 'Validation acc', "precision", "recall"])
            data = pd.concat([data, _df], ignore_index=True)
    data.to_csv(f'output/model_comparison/cnn_generalization_{min_cars}_{max_cars}.csv')