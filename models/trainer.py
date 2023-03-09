import copy
import json
import time

import numpy as np
import pandas as pd
import timm
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from matplotlib import pyplot as plt
from numpy import arange
from rtpt.rtpt import RTPT
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, precision_score, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit, KFold, ShuffleSplit
from tabulate import tabulate
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from itertools import product

from torchvision.models import ResNet18_Weights, ResNet50_Weights, ResNet101_Weights

from michalski_trains.dataset import get_datasets
from models.mlp.mlp import MLP
from models.multi_label_nn import MultiLabelNeuralNetwork, print_train, show_torch_im
from models.multioutput_regression.pos_net import PositionNetwork
from models.set_transformer import SetTransformer
from models.spacial_attr_net.attr_net import AttributeNetwork
from util import *
from visualization.vis_model import visualize_statistics, vis_confusion_matrix
from visualization.vis_model_comparison import model_scene_imcount_comparison, csv_to_tex_table


class Trainer:
    def __init__(self, base_scene, train_col, train_vis, device, model_name, class_rule, ds_path,
                 X_val='image', y_val='direction', max_car=4, min_car=2,
                 resume=False, pretrained=True, resize=False, optimizer_='ADAM', loss='CrossEntropyLoss',
                 train_samples=10000, ds_size=10000, image_noise=0, label_noise=0,
                 batch_size=50, num_worker=4, lr=0.001, step_size=5, gamma=.8, momentum=0.9,
                 num_epochs=25, setup_model=True, setup_ds=True, save_model=True):

        # ds_val setup
        self.base_scene, self.train_col, self.train_vis, self.class_rule = base_scene, train_col, train_vis, class_rule
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
        self.model = self.setup_model(resume=resume) if setup_model else None
        self.full_ds = get_datasets(base_scene, self.train_col, self.train_vis, ds_size=ds_size, ds_path=ds_path,
                                    y_val=y_val, max_car=self.max_car, min_car=self.min_car,
                                    class_rule=class_rule, resize=resize,
                                    preprocessing=self.preprocess) if setup_ds else None

    def cross_val_train(self, train_size=None, label_noise=None, rules=None, visualizations=None, scenes=None,
                        n_splits=5, model_path=None, save_models=False, replace=False, image_noise=None):
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
        tr_it = 0
        tr_max = n_splits * len(train_size) * len(label_noise) * len(image_noise) * len(rules) * len(
            visualizations) * len(scenes)
        for l_noise, i_noise, rule, visualization, scene in product(label_noise, image_noise, rules, visualizations,
                                                                    scenes):
            self.label_noise, self.image_noise, self.class_rule, self.train_vis, self.base_scene = l_noise, i_noise, rule, visualization, scene
            self.full_ds = get_datasets(self.base_scene, self.train_col, self.train_vis, class_rule=rule,
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
                    if not (os.path.isfile(self.out_path + 'metrics.json') and os.path.isfile(
                            self.out_path + 'model.pth')) or replace:
                        print('====' * 10)
                        print(f'training iteration {tr_it} of {tr_max}')
                        self.setup_model(resume=self.resume, path=model_path)
                        self.setup_ds(tr_idx=tr_idx, val_idx=val_idx)
                        self.train(rtpt_extra=(tr_max - tr_it) * self.num_epochs, set_up=False)
                        del self.model
                    tr_it += 1

    def train(self, rtpt_extra=0, ds_size=None, set_up=True):
        if self.full_ds is None:
            self.full_ds = get_datasets(self.base_scene, self.train_col, self.train_vis, class_rule=self.class_rule,
                                        ds_size=self.ds_size, max_car=self.max_car, min_car=self.min_car,
                                        label_noise=self.label_noise, image_noise=self.image_noise,
                                        ds_path=self.ds_path,
                                        resize=self.resize)
        if set_up:
            self.ds_size = ds_size if ds_size is not None else self.ds_size
            self.setup_model(self.resume)
            self.setup_ds()
        self.model = do_train(self.base_scene, self.train_col, self.y_val, self.device, self.out_path, self.model_name,
                              self.model, self.full_ds, self.dl, self.checkpoint, self.optimizer, self.scheduler,
                              self.criteria, num_epochs=self.num_epochs, lr=self.lr, step_size=self.step_size,
                              gamma=self.gamma, save_model=self.save_model, rtpt_extra=rtpt_extra
                              )
        torch.cuda.empty_cache()

    def val(self, val_size=None, set_up=True, model_path=None):
        eps = self.num_epochs
        self.num_epochs = 1
        if set_up:
            val_size = val_size if val_size is not None else self.ds_size
            self.setup_ds(val_size=val_size)
            self.setup_model(self.resume, path=model_path)
        acc, precision, recall = do_train(self.base_scene, self.train_col, self.y_val, self.device, self.out_path,
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
        self.model = self.get_model(self.model_name, self.pretrained, dim_out, class_dim)

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])

        if self.optimizer_name == 'SGD':
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.lr, momentum=self.momentum)
        elif self.optimizer_name == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-5)
        else:
            raise AssertionError('specify valid optimizer')
        if self.checkpoint is not None:
            self.optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])

        # apparently the optimizer is not on gpu -> send it to the gpu
        optimizer_to(self.optimizer, device=self.device)

        # self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=self.step_size, gamma=self.gamma)

    def setup_ds(self, tr_idx=None, val_idx=None, train_size=None, val_size=None):
        if tr_idx is None or val_idx is None:
            if train_size is None and val_size is None:
                train_size = int(0.8 * self.ds_size)
                val_size = int(0.2 * self.ds_size)
            elif train_size is None:
                train_size = 0
            else:
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
        self.dl = {
            'train': DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_worker),
            'val': DataLoader(self.ds['val'], batch_size=self.batch_size, num_workers=self.num_worker)
        }

    def get_model(self, model_name, pretrained, num_output, num_class):
        if model_name == 'resnet18':
            model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        elif model_name == 'resnet50':
            model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == 'resnet101':
            model = models.resnet101(weights=ResNet101_Weights.DEFAULT)
        elif model_name == 'VisionTransformer':
            model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=2)
        elif model_name == 'EfficientNet':
            # model = models.efficientnet_b7(pretrained=pretrained, num_classes=2)
            # model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=2)
            # model = timm.create_model('tf_efficientnet_b7_ns', pretrained=pretrained, num_classes=2)
            model = timm.create_model('tf_efficientnetv2_l_in21k', pretrained=pretrained, num_classes=2)
        elif model_name == 'set_transformer':
            model = SetTransformer(dim_input=32, dim_output=num_output * num_class)
        elif model_name == 'faster_rcnn':
            weights = models.detection.FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
            model = models.detection.fasterrcnn_resnet50_fpn_v2(weights=weights, box_score_thresh=0.9)
            self.preprocess = weights.transforms()
        elif model_name == 'attr_predictor':
            model = AttributeNetwork(dim_input=32)
        elif model_name == 'pos_predictor':
            model = PositionNetwork(dim_input=4, dim_output=num_output)
        elif model_name == 'MLP':
            model = MLP(dim_in=4 * 32, dim_out=num_output * num_class)
        else:
            raise AssertionError('select valid model')

        if 'resnet' in model_name:
            if num_class == 2:
                num_ftrs = model.fc.in_features
                # Here the size of each output sample is set to 2.
                # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
                model.fc = nn.Linear(num_ftrs, num_class)
            else:
                model = MultiLabelNeuralNetwork(model, num_output)
        return model

    def plt_accuracy(self):
        visualize_statistics(self.train_col, self.base_scene, self.y_val, self.ds, self.out_path, self.model_name)

    def plt_confusion_matrix(self):
        vis_confusion_matrix(self.train_col, self.base_scene, self.out_path, self.model_name, self.model,
                             self.dl, self.device)

    def plt_cross_val_performance(self, tex_table=False, models=None):
        if models is None:
            models = [self.model_name]
        print('plotting  cross validated performance')
        path = self.get_model_path()
        model_scene_imcount_comparison(self.train_col, models, self.y_val, path)
        if tex_table:
            csv_to_tex_table(path + 'mean_variance_comparison.csv')

    def get_model_path(self, prefix=False, suffix='', im_count=None, model_name=None):
        model_name = self.model_name if model_name is None else model_name
        pre = '_pretrained' if self.pretrained else ''
        im_count = self.ds_size if im_count is None else im_count
        ds_settings = f'{self.train_vis}_{self.class_rule}_{self.train_col}_{self.base_scene}'
        train_config = f'imcount_{im_count}_X_val_{self.X_val}{pre}_lr_{self.lr}_step_{self.step_size}_gamma{self.gamma}'
        pref = '' if not prefix else 'cv'
        if self.label_noise > 0:
            pref += f'_{self.label_noise}noise'
        if self.image_noise > 0:
            pref += f'_{self.image_noise}im_noise'
        pref += '/'

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

    def transfer_classification(self, train_size, n_splits=5, batch_size=None):
        print(f'transfer classification: {self.model_name} trained on base scene to predict other scenes')
        data = pd.DataFrame(columns=['methods', 'number of images', 'scenes', 'mean', 'variance', 'std'])
        data_cv = pd.DataFrame(columns=['Methods', 'number of images', 'cv iteration', 'Validation acc', 'scene'])
        train_size = [100, 1000, 8000] if train_size is None else train_size
        batch_size = self.batch_size if batch_size is None else batch_size

        rtpt = RTPT(name_initials='LH', experiment_name=f'trans', max_iterations=n_splits * 4 * len(train_size))
        rtpt.start()

        for scene in ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']:
            ds = get_datasets(scene, self.train_col, self.train_vis, 10000, resize=False,
                              ds_path=self.ds_path)
            dl = DataLoader(ds, batch_size=batch_size, num_workers=self.num_worker)
            for training_size in train_size:
                accs = []
                for fold in range(n_splits):
                    rtpt.step()
                    torch.cuda.memory_summary(device=None, abbreviated=False)

                    self.out_path = self.get_model_path(prefix=f'cv/', suffix=f'it_{fold}/')
                    del self.model
                    self.setup_model(resume=True)
                    self.model.eval()

                    all_labels = np.empty(0, int)
                    all_preds = np.empty(0, int)
                    # Iterate over data.
                    for inputs, labels in dl:
                        print(torch.cuda.memory_summary(device=None, abbreviated=False))
                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        self.model.to(self.device)
                        labels = torch.t(labels)
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        print(torch.cuda.memory_summary(device=None, abbreviated=False))

                        outputs = self.model(inputs)
                        if outputs.dim() < 3:
                            outputs = outputs.unsqueeze(dim=1)
                        outputs = torch.moveaxis(outputs, 0, 1)

                        preds = torch.max(outputs, dim=2)[1]

                        labels, preds = labels.to("cpu"), preds.to("cpu")
                        labels, preds = labels.detach().numpy(), preds.detach().numpy()
                        all_labels = np.hstack((all_labels, labels.flatten()))
                        all_preds = np.hstack((all_preds, preds.flatten()))
                    acc = accuracy_score(all_labels, all_preds) * 100
                    accs.append(acc)

                    print(f'{self.model_name} trained on base scene with {training_size} images (cv iteration {fold})'
                          f' achieves an accuracy of {acc} when classifying {scene} images')
                    li = [self.model_name, training_size, fold, acc, scene]
                    _df = pd.DataFrame([li], columns=['Methods', 'number of images', 'cv iteration', 'Validation acc',
                                                      'scene'])
                    data_cv = pd.concat([data_cv, _df], ignore_index=True)
                mean = sum(accs) / len(accs)
                variance = sum((xi - mean) ** 2 for xi in accs) / len(accs)
                std = np.sqrt(variance)
                li = [self.model_name, training_size, scene, mean, variance, std]
                _df = pd.DataFrame([li], columns=['methods', 'number of images', 'scenes', 'mean', 'variance', 'std'])
                data = pd.concat([data, _df], ignore_index=True)
        print(tabulate(data, headers='keys', tablefmt='psql'))
        path = f'output/models/{self.model_name}/{self.y_val}_classification/{self.train_col}/{self.base_scene}/'
        os.makedirs(path, exist_ok=True)
        data.to_csv(path + 'transfer_classification.csv')
        data_cv.to_csv(path + 'transfer_classification_cv.csv')
        # csv_to_tex_table(path + 'transfer_classification.csv', )

    def predict_train_description(self, use_transfer_trained_model=False, im_counts=None):

        # model_pred1 = 'attr_predictor'
        # model_pred2 = 'resnet18'
        # out_path = f'output/models/{model_pred2}/attribute_classification/RandomTrains/{self.base_scene}'
        # config = f'imcount_10000_X_val_predicted_mask_lr_0.001_step_5_gamma0.8'
        train_col = self.train_col
        dl = DataLoader(self.full_ds, batch_size=self.batch_size, shuffle=False, num_workers=4)
        if im_counts is None: im_counts = [100, 1000, 8000]
        if not use_transfer_trained_model:
            self.train_col = 'RandomTrains'
            im_counts = [8000]

        # self.X_val = 'gt_mask'

        for im_count in im_counts:
            out_path = f'output/models/{self.model_name}/attribute_classification/{self.train_col}/{self.base_scene}/predicted_descriptions/{im_count}'

            print(
                f'{self.model_name} trained on {im_count}{self.train_col} images predicting train descriptions for the'
                f' {train_col} trains in {self.base_scene}')
            accs = []
            for fold in range(5):

                path = self.get_model_path(prefix=f'cv/', suffix=f'it_{fold}/')
                self.setup_model(path=path, resume=True)

                self.model.eval()  # Set model to evaluate mode
                self.model.to(self.device)

                rtpt = RTPT(name_initials='LH', experiment_name=f'Pred_desc_{self.base_scene[:3]}',
                            max_iterations=self.full_ds.__len__() / self.batch_size)
                rtpt.start()

                with torch.no_grad():

                    all_labels = np.empty([0, 32], dtype=int)
                    all_preds = np.empty([0, 32], dtype=int)

                    # Iterate over data.
                    for inputs, labels in dl:
                        # for item in range(self.full_ds.__len__()):

                        inputs = inputs.to(self.device)
                        labels = labels.to(self.device)
                        self.model.to(self.device)

                        outputs = self.model(inputs)
                        if outputs.dim() < 3:
                            outputs = outputs.unsqueeze(dim=1)

                        preds = torch.max(outputs, dim=2)[1]

                        labels, preds = labels.to("cpu"), preds.to("cpu")
                        labels, preds = labels.detach().numpy(), preds.detach().numpy()

                        all_labels = np.vstack((all_labels, labels))
                        all_preds = np.vstack((all_preds, preds))
                        rtpt.step()
                acc = accuracy_score(all_labels.flatten(), all_preds.flatten())
                print(f'fold {fold} acc score: {acc}')
                os.makedirs(out_path, exist_ok=True)
                # print('acc score: ' + str(acc))
                accs.append(acc)

                np.save(out_path + f'/fold_{fold}.npy', all_preds, allow_pickle=True)
                del self.model
            print('average acc score: ' + str(np.mean(accs)))


def do_train(base_scene, train_col, y_val, device, out_path, model_name, model, full_ds, dl,
             checkpoint, optimizer, scheduler, criteria, num_epochs=25, lr=0.001, step_size=5, gamma=.8,
             save_model=True, rtpt_extra=0, train='train'):
    rtpt = RTPT(name_initials='LH', experiment_name=f'train_{base_scene[:3]}_{train_col[0]}',
                max_iterations=num_epochs + rtpt_extra)
    rtpt.start()
    epoch_init = 0
    phases = ['train', 'val'] if train == 'train' else ['val']
    if train == 'train':
        print(f'{train} settings: {out_path}')

    if checkpoint is not None:
        epoch_init = checkpoint['epoch']
        loss = checkpoint['loss']

    label_names = full_ds.get_ds_labels()
    class_names = full_ds.get_ds_classes()
    unique_label_names = list(set(label_names))
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    dataset_sizes = {
        'train': len(dl['train'].dataset),
        'val': len(dl['val'].dataset)
    }

    epoch_loss = {
        'train': [0] * num_epochs,
        'val': [0] * num_epochs
    }
    performance_metrics = ['bal_acc', 'acc', 'precision', 'recall']
    epoch_label_accs = {}
    epoch_acum_accs = {}
    for phase in ['train', 'val']:
        epoch_acum_accs[phase] = {}
        for metric in ['bal_acc', 'acc']:
            epoch_acum_accs[phase][metric] = [0] * num_epochs
        epoch_label_accs[phase] = {}

        for label in label_names:
            epoch_label_accs[phase][label] = {}
            for metric in ['bal_acc', 'acc']:
                epoch_label_accs[phase][label][metric] = [0] * num_epochs

        for metric in ['precision', 'recall']:
            epoch_label_accs[phase][metric] = {}
            for l_class in class_names:
                epoch_label_accs[phase][metric][l_class] = [0] * num_epochs

    for epoch in range(num_epochs):
        rtpt.step()
        time_elapsed = time.time() - since

        # Each epoch has a training and validation phase
        for phase in phases:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            all_labels = np.empty((len(unique_label_names), 0), int)
            all_preds = np.empty((len(unique_label_names), 0), int)

            # Iterate over data.
            for inputs, labels in dl[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                model.to(device)
                # input_labes = (inputs.amax(dim=(2, 3)) * 22).round().type(torch.int)
                # assert (input_labes - labels).sum() == 0
                labels = torch.t(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    if outputs.dim() < 3:
                        # if model only has 1 output add dimension
                        outputs = outputs.unsqueeze(dim=1)
                    # move multi output axis to front
                    outputs = torch.moveaxis(outputs, 0, 1)
                    # preds = torch.stack([torch.max(output, 1)[1] for output in outputs]).to(device)
                    preds = torch.max(outputs, dim=2)[1]

                    losses = [criterion(output, label) for label, output, criterion in zip(labels, outputs, criteria)]
                    loss = sum(losses)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                # print_train(outputs)
                # show_torch_im(inputs)

                # to numpy
                labels, preds = labels.to("cpu"), preds.to("cpu")
                labels, preds = labels.detach().numpy(), preds.detach().numpy()
                # print_train(labels)
                # print_train(preds)
                # combine the same attributes (labels) to a collective list e.g. color of car 3 and 4
                for i in range(len(label_names) // len(unique_label_names)):
                    all_labels = np.hstack(
                        (all_labels, labels[len(unique_label_names) * i:len(unique_label_names) * (i + 1)]))
                    all_preds = np.hstack(
                        (all_preds, preds[len(unique_label_names) * i:len(unique_label_names) * (i + 1)]))
                # statistics
                running_loss += loss.item() * inputs.size(0) / labels.shape[0]

            if phase == 'train':
                scheduler.step()
            epoch_loss[phase][epoch] = running_loss / dataset_sizes[phase]

            # recall = recall_score(all_labels.flatten(), all_preds.flatten(), average=None, zero_division=0)
            # precision = precision_score(all_labels.flatten(), all_preds.flatten(), average=None, zero_division=0)
            cm = confusion_matrix(all_labels.flatten(), all_preds.flatten(), )
            FP = cm.sum(axis=0) - np.diag(cm)
            FN = cm.sum(axis=1) - np.diag(cm)
            TP = np.diag(cm)
            TN = cm.sum() - (FP + FN + TP)
            recall = TP / (TP + FN)
            precision = TP / (TP + FP)
            for re, pre, class_name in zip(recall, precision, class_names):
                epoch_label_accs[phase]['recall'][class_name][epoch] = re
                epoch_label_accs[phase]['precision'][class_name][epoch] = pre

            for label, pred, label_name in zip(all_labels, all_preds, label_names):
                bal_acc = balanced_accuracy_score(label, pred)
                acc = accuracy_score(label, pred)
                for acc_type, metric_name in zip([bal_acc, acc], ['bal_acc', 'acc']):
                    epoch_label_accs[phase][label_name][metric_name][epoch] += acc_type
                    epoch_acum_accs[phase][metric_name][epoch] += acc_type / len(unique_label_names)
            # deep copy the model
            if phase == 'val' and epoch_acum_accs[phase]['acc'][epoch] > best_acc:
                best_acc = epoch_acum_accs[phase]['acc'][epoch]
                best_model_wts = copy.deepcopy(model.state_dict())
        if 'train' in phases:
            print(f'{model_name} training Epoch {epoch + 1}/{num_epochs}, ' +
                  f'elapsed time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s, ' +
                  f'val Loss: {round(epoch_loss["val"][epoch], 4)} Acc: {round(epoch_acum_accs["val"]["acc"][epoch] * 100, 2)}%, ' +
                  f'train Loss: {round(epoch_loss["train"][epoch], 4)} Acc: {round(epoch_acum_accs["train"]["acc"][epoch] * 100, 2)}%')

    os.makedirs(out_path, exist_ok=True)
    print_text = f'Best val Acc: {round(100 * best_acc, 2)}%'
    if train == 'train':
        print(print_text)
        # load best model weights
        model.load_state_dict(best_model_wts)
        statistics = {
            'epoch_label_accs': epoch_label_accs,
            'epoch_acum_accs': epoch_acum_accs,
            'epoch_loss': epoch_loss
        }

        if save_model:
            torch.save({
                'epoch': num_epochs + epoch_init,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, out_path + 'model.pth')

        if checkpoint is not None and os.path.isfile(out_path + 'metrics.json'):
            with open(out_path + 'metrics.json', 'r') as fp:
                stat_init = json.load(fp)
                statistics = merge(stat_init.copy(), statistics)
        with open(out_path + 'metrics.json', 'w+') as fp:
            json.dump(statistics, fp)
    elif train == 'val':
        print_text += f', TP: {TP[0]} TN: {TN[0]} FP: {FP[0]} FN: {FN[0]}, precision: {TP[0] / (TP[0] + FP[0])},' \
                      f' recall: {TP[0] / (TP[0] + FN[0])}'
        print(print_text)
        print('-' * 10)
        return best_acc, precision, recall
    print('-' * 10)
    return model.to('cpu')


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
