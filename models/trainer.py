import warnings
from itertools import product

import torch.nn as nn
import torch.optim as optim
from numpy import arange
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Subset

from michalski_trains.dataset import get_datasets
from models.model import get_model
from models.rcnn import rcnn_parallel
from models.rcnn.rcnn_train import train_rcnn
from models.train_loop import do_train
from util import *
from visualization.vis_model import visualize_statistics, vis_confusion_matrix
from visualization.vis_model_comparison import model_scene_imcount_comparison, csv_to_tex_table
from torch.utils.data.distributed import DistributedSampler


class Trainer:
    def __init__(self, base_scene, raw_trains, train_vis, device, model_name, class_rule, ds_path,
                 X_val='image', y_val='direction', max_car=4, min_car=2,
                 resume=False, pretrained=True, resize=False, optimizer_='ADAM', loss='CrossEntropyLoss',
                 train_size=None, val_size=None, ds_size=10000, image_noise=0, label_noise=0,
                 batch_size=50, num_worker=4, lr=0.001, step_size=5, gamma=.8, momentum=0.9,
                 num_epochs=25, setup_model=True, setup_ds=True, save_model=True):

        # ds_val setup
        self.settings = f'{train_vis}_{class_rule}_{raw_trains}_{base_scene}_len_{min_car}-{max_car}'

        self.base_scene, self.raw_trains, self.train_vis, self.class_rule = base_scene, raw_trains, train_vis, class_rule
        self.ds_path, self.ds_size = ds_path, ds_size
        self.max_car, self.min_car = max_car, min_car
        self.device = device
        self.X_val, self.y_val = X_val, y_val
        self.pretrained, self.resume, self.save_model = pretrained, resume, save_model
        self.resize, self.image_noise, self.label_noise = resize, image_noise, label_noise
        # model setup
        self.model_name = model_name
        self.optimizer_name, self.loss_name = optimizer_, loss
        # preprocessing needed for faster rcnn
        self.preprocess = None
        # training hyper parameter
        self.batch_size, self.num_worker, self.lr, self.step_size, self.gamma, self.momentum, self.num_epochs = \
            batch_size, num_worker, lr, step_size, gamma, momentum, num_epochs
        self.out_path = self.get_model_path()
        # setup model and dataset
        if setup_model:
            self.setup_model(resume=resume)
        else:
            self.model = None
        if setup_ds:
            self.full_ds = get_datasets(self.base_scene, self.raw_trains, self.train_vis, ds_size=self.ds_size,
                                        ds_path=self.ds_path,
                                        y_val=y_val, max_car=self.max_car, min_car=self.min_car,
                                        class_rule=self.class_rule,
                                        resize=self.resize, preprocessing=self.preprocess)
            self.setup_ds(train_size, val_size)
        else:
            self.full_ds = None

    def cross_val_train(self, train_size=None, label_noise=None, rules=None, visualizations=None, scenes=None,
                        n_splits=5, model_path=None, save_models=False, replace=False, image_noise=None, ex_name=None,
                        start_it=0):
        if train_size is None:
            train_size = [10000]
        if label_noise is None:
            label_noise = [self.label_noise]
        if image_noise is None:
            image_noise = [self.image_noise]
        if rules is None:
            rules = [self.class_rule]
        if visualizations is None:
            visualizations = [self.train_vis]
        if scenes is None:
            scenes = [self.base_scene]
        random_state = 0
        test_size = 2000
        self.save_model = save_models
        tr_it, tr_b = 0, 0
        n_batches = sum(train_size) // self.batch_size
        tr_b_total = n_splits * n_batches * len(label_noise) * len(image_noise) * len(rules) * len(
            visualizations) * len(scenes)
        tr_it_total = n_splits * len(train_size) * len(label_noise) * len(image_noise) * len(rules) * len(
            visualizations) * len(scenes)
        for l_noise, i_noise, rule, visualization, scene in product(label_noise, image_noise, rules, visualizations,
                                                                    scenes):
            self.label_noise, self.image_noise, self.class_rule, self.train_vis, self.base_scene = l_noise, i_noise, rule, visualization, scene
            self.full_ds = get_datasets(self.base_scene, self.raw_trains, self.train_vis, class_rule=rule,
                                        ds_size=self.ds_size, max_car=self.max_car, min_car=self.min_car,
                                        label_noise=self.label_noise, image_noise=self.image_noise,
                                        ds_path=self.ds_path,
                                        resize=self.resize)
            for t_size in train_size:
                self.full_ds.predictions_im_count = t_size
                cv = StratifiedShuffleSplit(train_size=t_size, test_size=test_size, random_state=random_state,
                                            n_splits=n_splits)
                y = np.concatenate([self.full_ds.get_direction(item) for item in range(self.full_ds.__len__())])

                for fold, (tr_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
                    self.out_path = self.get_model_path(prefix=True, suffix=f'it_{fold}/', im_count=t_size)
                    if tr_it >= start_it and (not (os.path.isfile(self.out_path + 'metrics.json') and os.path.isfile(
                            self.out_path + 'model.pth')) or replace):
                        print('====' * 10)
                        print(
                            f'training iteration {tr_it + 1}/{tr_it_total} with {t_size // self.batch_size} '
                            f'training batches, already completed: {tr_b}/{tr_b_total} batches. ')
                        self.setup_model(resume=self.resume, path=model_path)
                        self.setup_ds(tr_idx=tr_idx, val_idx=val_idx)
                        self.train(rtpt_extra=(tr_b_total - tr_b) * self.num_epochs, set_up=False, ex_name=ex_name)
                        del self.model
                    tr_b += t_size // self.batch_size
                    tr_it += 1

    def train(self, rtpt_extra=0, train_size=None, val_size=None, set_up=True, ex_name=None, gpu_count=1):
        if self.full_ds is None:
            self.full_ds = get_datasets(self.base_scene, self.raw_trains, self.train_vis, class_rule=self.class_rule,
                                        ds_size=self.ds_size, max_car=self.max_car, min_car=self.min_car,
                                        label_noise=self.label_noise, image_noise=self.image_noise,
                                        ds_path=self.ds_path, y_val=self.y_val,
                                        resize=self.resize)
        if set_up:
            # self.ds_size = ds_size if ds_size is not None else self.ds_size
            self.setup_model(self.resume)
            self.setup_ds(train_size=train_size, val_size=val_size)
        if self.model_name == 'rcnn':
            if gpu_count >= 1:
                rcnn_parallel.train_parallel(self.out_path, self.model, self.ds, self.optimizer, self.scheduler,
                                             self.num_epochs, self.batch_size, self.save_model, world_size=gpu_count,
                                             ex_name=ex_name)
            else:
                self.model = train_rcnn(self.base_scene, self.raw_trains, self.y_val, self.device, self.out_path,
                                        self.model_name, self.model, self.full_ds, self.dl, self.checkpoint,
                                        self.optimizer,
                                        self.scheduler, self.criteria, num_epochs=self.num_epochs, lr=self.lr,
                                        step_size=self.step_size, gamma=self.gamma, save_model=self.save_model,
                                        rtpt_extra=rtpt_extra, ex_name=ex_name
                                        )
        else:
            self.model = do_train(self.base_scene, self.raw_trains, self.y_val, self.device, self.out_path,
                                  self.model_name, self.model, self.full_ds, self.dl, self.checkpoint, self.optimizer,
                                  self.scheduler, self.criteria, num_epochs=self.num_epochs, lr=self.lr,
                                  step_size=self.step_size, gamma=self.gamma, save_model=self.save_model,
                                  rtpt_extra=rtpt_extra, ex_name=ex_name
                                  )
        torch.cuda.empty_cache()

    def val(self, val_size=None, set_up=True, model_path=None):
        eps = self.num_epochs
        self.num_epochs = 1
        if set_up:
            val_size = val_size if val_size is not None else self.ds_size
            self.setup_ds(val_size=val_size)
            self.setup_model(self.resume, path=model_path)
        if self.model_name == 'rcnn':
            acc, precision, recall = train_rcnn(self.base_scene, self.raw_trains, self.y_val, self.device,
                                                self.out_path,
                                                self.model_name,
                                                self.model, self.full_ds, self.dl, self.checkpoint, self.optimizer,
                                                self.scheduler,
                                                self.criteria, num_epochs=self.num_epochs, lr=self.lr,
                                                step_size=self.step_size,
                                                gamma=self.gamma, save_model=False, train='val')

        else:
            acc, precision, recall = do_train(self.base_scene, self.raw_trains, self.y_val, self.device, self.out_path,
                                              self.model_name,
                                              self.model, self.full_ds, self.dl, self.checkpoint, self.optimizer,
                                              self.scheduler,
                                              self.criteria, num_epochs=self.num_epochs, lr=self.lr,
                                              step_size=self.step_size,
                                              gamma=self.gamma, save_model=False, train='val')
        torch.cuda.empty_cache()
        self.num_epochs = eps
        return acc, precision, recall

    def setup_model(self, resume=False, path=None):
        # set path
        path = self.out_path if path is None else path
        set_up_txt = f'set up {self.model_name}'

        if resume and os.path.isfile(path + 'model.pth') and os.path.isfile(path + 'metrics.json'):
            self.checkpoint = torch.load(path + 'model.pth', map_location=self.device)
            set_up_txt += ': loaded from ' + path
        elif resume and os.path.isfile(path + 'model.pth'):
            self.checkpoint = torch.load(path + 'model.pth', map_location=self.device)
            set_up_txt += ': loaded from ' + path
            warnings.warn('no metrics found')
        else:  # no checkpoint found
            if resume:
                raise AssertionError(f'no pretrained model or metrics found at {path}\n please train model first')
            self.checkpoint = None

        print(set_up_txt)
        dim_out = get_output_dim(self.y_val)
        class_dim = get_class_dim(self.y_val)
        if self.loss_name == 'MSELoss':
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
        self.criteria = [loss_fn] * dim_out
        self.model, self.preprocess = get_model(self.model_name, self.pretrained and not resume, dim_out, class_dim)

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])

        if self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer_name == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        elif self.optimizer_name == 'ADAMW':
            self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        else:
            raise AssertionError('specify valid optimizer')
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        # apparently the optimizer is not on gpu -> send it to the gpu
        optimizer_to(self.optimizer, device=self.device)

        # self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def setup_ds(self, tr_idx=None, val_idx=None, train_size=None, val_size=None):
        if self.full_ds is None:
            self.full_ds = get_datasets(self.base_scene, self.raw_trains, self.train_vis, ds_size=self.ds_size,
                                        ds_path=self.ds_path, y_val=self.y_val, max_car=self.max_car,
                                        min_car=self.min_car, class_rule=self.class_rule, resize=self.resize,
                                        preprocessing=self.preprocess)
        if tr_idx is None or val_idx is None:
            if train_size is None and val_size is None:
                train_size = int(0.8 * self.ds_size)
                val_size = int(0.2 * self.ds_size)
            elif train_size is None:
                train_size = 0
            elif val_size is None:
                val_size = int(0.2 * self.ds_size)
            tr_idx = arange(train_size)
            val_idx = arange(train_size, train_size + val_size)
        if len(tr_idx) > 0:
            set_up_txt = f'setup ds with {len(tr_idx)} images for training and {len(val_idx)} images for validation'
            print(set_up_txt)

        self.ds = {
            'train': Subset(self.full_ds, tr_idx),
            'val': Subset(self.full_ds, val_idx)
        }
        if self.model_name == 'rcnn':
            self.dl = {'train': DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_worker,
                                           collate_fn=collate_fn_rcnn),
                       'val': DataLoader(self.ds['val'], batch_size=self.batch_size, num_workers=self.num_worker,
                                         collate_fn=collate_fn_rcnn)}
        else:
            self.dl = {'train': DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_worker),
                       'val': DataLoader(self.ds['val'], batch_size=self.batch_size, num_workers=self.num_worker)}

    def plt_accuracy(self):
        visualize_statistics(self.raw_trains, self.base_scene, self.y_val, self.ds, self.out_path, self.model_name)

    def plt_confusion_matrix(self):
        vis_confusion_matrix(self.raw_trains, self.base_scene, self.out_path, self.model_name, self.model,
                             self.dl, self.device)

    def plt_cross_val_performance(self, tex_table=False, models=None):
        if models is None:
            models = [self.model_name]
        print('plotting  cross validated performance')
        path = self.get_model_path()
        model_scene_imcount_comparison(self.raw_trains, models, self.y_val, path)
        if tex_table:
            csv_to_tex_table(path + 'mean_variance_comparison.csv')

    def get_model_path(self, prefix=False, suffix='', im_count=None, model_name=None):
        model_name = self.model_name if model_name is None else model_name
        pre = '_pretrained' if self.pretrained else ''
        im_count = self.ds_size if im_count is None else im_count
        ds_settings = f'{self.train_vis}_{self.class_rule}_{self.raw_trains}_{self.base_scene}'
        train_config = f'imcount_{im_count}_X_val_{self.X_val}{pre}_lr_{self.lr}_step_{self.step_size}_gamma{self.gamma}'
        pref = '' if not prefix else 'cv'
        if self.label_noise > 0:
            pref += f'_{self.label_noise}noise'
        if self.image_noise > 0:
            pref += f'_{self.image_noise}im_noise'
        pref = pref if pref == '' else f'{pref}/'

        # if self.label_noise > 0:
        #     train_config += f'noise_{self.label_noise}'
        # if self.image_noise > 0:
        #     train_config += f'im_noise_{self.image_noise}'
        # if prefix and self.label_noise > 0:
        #     pref = f'cv_{self.label_noise}noise/'
        # elif prefix:
        #     pref = 'cv/'
        # else:
        #     pref = ''
        out_path = f'output/models/{model_name}/{self.y_val}_classification/{ds_settings}/{pref}{train_config}/{suffix}'
        return out_path


# copied from https://discuss.pytorch.org/t/moving-optimizer-from-cpu-to-gpu/96068
def optimizer_to(optim, device):
    for param in optim.state.values():
        # Not sure there are any global tensors in the state dict
        if isinstance(param, torch.Tensor):
            param.data = param.data.to(device)
            if param._grad is not None:
                param._grad.data = param._grad.data.to(device)
        elif isinstance(param, dict):
            for subparam in param.values():
                if isinstance(subparam, torch.Tensor):
                    subparam.data = subparam.data.to(device)
                    if subparam._grad is not None:
                        subparam._grad.data = subparam._grad.data.to(device)


def get_class_dim(y_val):
    '''
    Get the number of classes for each label
    :param y_val: type of y_val
    :return: number of classes for each label
    '''
    # ds labels
    labels = ['direction']
    label_classes = ['west', 'east']
    attributes = ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2',
                  'load_obj3'] * 4
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
    output = {
        'direction': len(label_classes),
        'attributes': len(attribute_classes),
        'mask': len(attribute_classes),
    }
    return output[y_val]


def get_class_names(y_val):
    '''
    Get the class names for each label
    :param y_val: type of y_val
    :return: class names for each label
    '''
    # ds labels
    labels = ['direction']
    label_classes = ['west', 'east']
    attributes = ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2',
                  'load_obj3'] * 4
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
    output = {
        'direction': label_classes,
        'attributes': attribute_classes,
        'mask': attribute_classes,
    }
    return output[y_val]


def get_output_dim(y_val):
    '''
    Get number of labels
    :param y_val: type of y_val
    :return: number of labels
    '''
    # ds labels
    labels = ['direction']
    label_classes = ['west', 'east']
    attributes = ['color', 'length', 'walls', 'roofs', 'wheel_count', 'load_obj1', 'load_obj2',
                  'load_obj3'] * 4
    color = ['yellow', 'green', 'grey', 'red', 'blue']
    length = ['short', 'long']
    walls = ["braced_wall", 'solid_wall']
    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof']
    wheel_count = ['2_wheels', '3_wheels']
    load_obj = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase']
    attribute_classes = ['none'] + color + length + walls + roofs + wheel_count + load_obj
    output = {
        'direction': len(labels),
        'attributes': len(attributes),
        'mask': len(attributes),
    }
    return output[y_val]


def collate_fn_rcnn(batch):
    """
    To handle the data loading as different images may have different number
    of objects and to handle varying size tensors as well.
    """
    return tuple(zip(*batch))
