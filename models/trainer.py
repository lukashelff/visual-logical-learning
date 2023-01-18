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
from sklearn.metrics import balanced_accuracy_score, accuracy_score, recall_score, precision_score
from sklearn.model_selection import StratifiedShuffleSplit, KFold, ShuffleSplit
from tabulate import tabulate
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.data import DataLoader
from itertools import product

from michalski_trains.m_train_dataset import get_datasets
from models.mlp.mlp import MLP
from models.multi_label_nn import MultiLabelNeuralNetwork
from models.multioutput_regression.pos_net import PositionNetwork
from models.set_transformer import SetTransformer
from models.spacial_attr_net.attr_net import AttributeNetwork
from util import *
from visualization.vis_model import visualize_statistics, vis_confusion_matrix
from visualization.vis_model_comparison import model_scene_imcount_comparison, csv_to_tex_table


class Trainer:
    def __init__(self, base_scene, train_col, train_vis, device, model_name, class_rule, ds_path,
                 X_val='image', y_val='direction',
                 resume=False, pretrained=True, resize=False, optimizer_='ADAM', loss='CrossEntropyLoss',
                 train_samples=10000, ds_size=10000, noise=0,
                 batch_size=50, num_worker=4, lr=0.001, step_size=5, gamma=.8, momentum=0.9,
                 num_epochs=25, setup_model=True, setup_ds=True, save_model=True):
        if y_val == 'direction' and train_col == 'RandomTrains':
            raise AssertionError(f'There is no direction label for a {train_col}. Use MichalskiTrain DS.')

        # ds_val setup
        self.base_scene, self.train_col, self.train_vis, self.class_rule = base_scene, train_col, train_vis, class_rule
        self.ds_path = ds_path
        self.device = device
        self.X_val, self.y_val = X_val, y_val
        self.pretrained, self.resume, self.save_model = pretrained, resume, save_model
        self.resize, self.noise = resize, noise
        # self.full_ds = get_datasets(self.base_scene, self.train_col, 10000, self.y_val, resize=resize,
        #                             X_val=self.X_val)
        self.full_ds = get_datasets(base_scene, self.train_col, self.train_vis, ds_size, ds_path=ds_path,
                                    class_rule=class_rule, resize=resize)
        # model setup
        self.model_name = model_name
        self.optimizer_name, self.loss_name = optimizer_, loss
        # training hyper parameter
        self.image_count, self.batch_size, self.num_worker, self.lr, self.step_size, self.gamma, self.momentum, self.num_epochs = \
            train_samples, batch_size, num_worker, lr, step_size, gamma, momentum, num_epochs
        self.out_path = self.update_out_path()
        if setup_model:
            self.setup_model(resume)
        if setup_ds:
            self.setup_ds()

    def cross_val_train(self, train_size=None, noises=None, n_splits=5, model_path=None, save_models=False,
                        replace=False):
        if train_size is None:
            train_size = [100, 1000, 10000]
        if noises is None:
            noises = [0, 0.1, 0.3]
        random_state = 0
        test_size = 2000
        self.save_model = save_models
        if self.train_col == 'MichalskiTrains':
            y = np.concatenate([self.full_ds.get_direction(item) for item in range(self.full_ds.__len__())])
        else:
            y = np.zeros(self.full_ds.__len__())
        rtpt_extra = n_splits * len(train_size) * self.num_epochs
        for training_size, noise in product(train_size, noises):
            self.image_count = training_size
            self.noise = noise
            self.full_ds.predictions_im_count = training_size
            if self.train_col == 'MichalskiTrains':
                cv = StratifiedShuffleSplit(train_size=training_size, test_size=test_size, random_state=random_state,
                                            n_splits=n_splits)
            else:
                cv = ShuffleSplit(n_splits=n_splits, train_size=training_size, test_size=test_size, )
            for fold, (tr_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
                self.out_path = self.update_out_path(prefix=True, suffix=f'it_{fold}/')
                self.setup_model(resume=self.resume, path=model_path)
                self.setup_ds(tr_idx=tr_idx, val_idx=val_idx)

                if not os.path.isdir(self.out_path) or replace:
                    self.train(rtpt_extra=rtpt_extra)
                rtpt_extra -= self.num_epochs
                del self.model

    def train(self, rtpt_extra=0):
        self.model = do_train(self.base_scene, self.train_col, self.y_val, self.device, self.out_path, self.model_name,
                              self.model, self.full_ds, self.dl, self.checkpoint, self.optimizer, self.scheduler,
                              self.criteria, num_epochs=self.num_epochs, lr=self.lr, step_size=self.step_size,
                              gamma=self.gamma, save_model=self.save_model, rtpt_extra=rtpt_extra
                              )
        torch.cuda.empty_cache()

    def setup_model(self, resume=False, path=None):
        # set path
        path = self.out_path if path is None else path
        set_up_txt = f'set up {self.model_name}'

        if resume and os.path.isfile(path + 'model.pth') and os.path.isfile(path + 'metrics.json'):
            self.checkpoint = torch.load(path + 'model.pth', map_location=self.device)
            set_up_txt += ': loaded from ' + path
        else:
            if resume:
                raise AssertionError(f'no pretrained model or metrics found at {path}\n please train model first')
            self.checkpoint = None

        print(set_up_txt)
        dim_out = self.full_ds.output_dim
        if self.loss_name == 'MSELoss':
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
        self.criteria = [loss_fn] * dim_out
        self.model = get_model(self.model_name, self.pretrained, dim_out, self.full_ds.class_dim)

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

    def setup_ds(self, tr_idx=None, val_idx=None):
        if tr_idx is None or val_idx is None:
            train_size, val_size = int(0.8 * self.image_count), int(0.2 * self.image_count)
            tr_idx = arange(train_size)
            val_idx = arange(train_size, train_size + val_size)
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

    def plt_accuracy(self):
        visualize_statistics(self.train_col, self.base_scene, self.y_val, self.ds, self.out_path, self.model_name)

    def plt_confusion_matrix(self):
        vis_confusion_matrix(self.train_col, self.base_scene, self.out_path, self.model_name, self.model,
                             self.dl, self.device)

    def plt_cross_val_performance(self, tex_table=False):
        print('plotting  cross validated performance')
        path = self.update_out_path()
        model_scene_imcount_comparison(self.train_col, [self.model_name], self.y_val, path)
        if tex_table:
            csv_to_tex_table(path + 'mean_variance_comparison.csv')

    def update_out_path(self, prefix=False, suffix='', im_count=None):
        pre = '_pretrained' if self.pretrained else ''
        im_count = self.image_count if im_count is None else im_count
        ds_settings = f'{self.train_vis}_{self.class_rule}_{self.train_col}_{self.base_scene}'
        train_config = f'imcount_{im_count}_X_val_{self.X_val}{pre}_lr_{self.lr}_step_{self.step_size}_gamma{self.gamma}'
        if self.noise > 0:
            train_config += f'noise_{self.noise}'
        if prefix and self.noise > 0:
            pref = f'cv_{self.noise}noise/'
        elif prefix:
            pref = 'cv/'
        else:
            pref = ''
        out_path = f'output/models/{self.model_name}/{self.y_val}_classification/{ds_settings}/{pref}{train_config}/{suffix}'
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
                self.image_count = training_size
                accs = []
                for fold in range(n_splits):
                    rtpt.step()
                    torch.cuda.memory_summary(device=None, abbreviated=False)

                    self.out_path = self.update_out_path(prefix=f'cv/', suffix=f'it_{fold}/')
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
                    li = [self.model_name, self.image_count, fold, acc, scene]
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

            self.image_count = im_count
            print(
                f'{self.model_name} trained on {im_count}{self.train_col} images predicting train descriptions for the'
                f' {train_col} trains in {self.base_scene}')
            accs = []
            for fold in range(5):

                path = self.update_out_path(prefix=f'cv/', suffix=f'it_{fold}/')
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
             save_model=True, rtpt_extra=0):
    rtpt = RTPT(name_initials='LH', experiment_name=f'train_{base_scene[:3]}_{train_col[0]}',
                max_iterations=num_epochs + rtpt_extra)
    rtpt.start()
    epoch_init = 0
    if checkpoint is not None:
        epoch_init = checkpoint['epoch']
        loss = checkpoint['loss']

    label_names = full_ds.get_ds_labels()
    unique_label_names = list(set(label_names))
    class_names = full_ds.label_classes
    num_classes = len(class_names)
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
        # Update the RTPT (subtitle is optional)
        rtpt.step()
        print('-' * 10)
        print(f'Epoch {epoch + 1}/{num_epochs} training {model_name}')
        print(f'settings {out_path}')
        time_elapsed = time.time() - since
        print('Elapsed time: {:.0f}m {:.0f}s'.format(
            time_elapsed // 60, time_elapsed % 60))

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
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

                # to numpy
                labels, preds = labels.to("cpu"), preds.to("cpu")
                labels, preds = labels.detach().numpy(), preds.detach().numpy()
                num_labels = len(unique_label_names)
                # combine the same attributes (labels) to a collective list e.g. color of car 3 and 4
                for i in range(len(label_names) // num_labels):
                    all_labels = np.hstack((all_labels, labels[num_labels * i:num_labels * (i + 1)]))
                    all_preds = np.hstack((all_preds, preds[num_labels * i:num_labels * (i + 1)]))
                # statistics
                running_loss += loss.item() * inputs.size(0) / labels.shape[0]
            if phase == 'train':
                scheduler.step()
            epoch_loss[phase][epoch] = running_loss / dataset_sizes[phase]

            recall = recall_score(all_labels.flatten(), all_preds.flatten(), average=None, zero_division=0)
            precision = precision_score(all_labels.flatten(), all_preds.flatten(), average=None, zero_division=0)
            for re, pre, class_name in zip(recall, precision, class_names):
                epoch_label_accs[phase]['recall'][class_name][epoch] = re
                epoch_label_accs[phase]['precision'][class_name][epoch] = pre

            for label, pred, label_name in zip(all_labels, all_preds, label_names):
                bal_acc = balanced_accuracy_score(label, pred)
                acc = accuracy_score(label, pred)
                for acc_type, metric_name in zip([bal_acc, acc], ['bal_acc', 'acc']):
                    epoch_label_accs[phase][label_name][metric_name][epoch] += acc_type
                    epoch_acum_accs[phase][metric_name][epoch] += acc_type / len(unique_label_names)
            print(
                '{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss[phase][epoch],
                                                     epoch_acum_accs[phase]['acc'][epoch]))
            # deep copy the model
            if phase == 'val' and epoch_acum_accs[phase]['acc'][epoch] > best_acc:
                best_acc = epoch_acum_accs[phase]['acc'][epoch]
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Acc: {:4f}'.format(best_acc))
    os.makedirs(out_path, exist_ok=True)

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
    return model.to('cpu')


def get_model(model_name, pretrained, num_output, num_class):
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    elif model_name == 'resnet101':
        model = models.resnet101(pretrained=pretrained)
    elif model_name == 'VisionTransformer':
        model = timm.create_model('vit_large_patch16_224', pretrained=pretrained, num_classes=2)
    elif model_name == 'EfficientNet':
        # model = models.efficientnet_b7(pretrained=pretrained, num_classes=2)
        # model = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=2)
        # model = timm.create_model('tf_efficientnet_b7_ns', pretrained=pretrained, num_classes=2)
        model = timm.create_model('tf_efficientnetv2_l_in21k', pretrained=pretrained, num_classes=2)
    elif model_name == 'set_transformer':
        model = SetTransformer(dim_input=32, dim_output=num_output * num_class)
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
