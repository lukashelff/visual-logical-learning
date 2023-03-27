import copy
import json
import math
import time

import numpy as np
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy import arange
from rtpt.rtpt import RTPT
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import StratifiedShuffleSplit
from torch.optim import lr_scheduler
from torch.utils.data import Subset
from torch.utils.data import DataLoader

from m_train_dataset import get_datasets
from models.trainer import optimizer_to, get_model
from util import *
from visualization.vis_model import visualize_statistics, vis_confusion_matrix


class Regression:
    def __init__(self, base_scene, raw_trains, device, model_name,
                 X_val='image', y_val='direction',
                 resume=False, pretrained=True, resize=False, optimizer_='ADAM', loss='CrossEntropyLoss',
                 image_count=10000, batch_size=50, num_worker=4, lr=0.001, step_size=5, gamma=.8, momentum=0.9,
                 num_epochs=25):
        if y_val == 'direction' and raw_trains == 'RandomTrains':
            raise AssertionError(f'There is no direction label for a {raw_trains}. Use MichalskiTrain DS.')

        # ds_val setup
        self.base_scene, self.raw_trains, self.device = base_scene, raw_trains, device
        self.X_val, self.y_val = X_val, y_val
        self.pretrained = pretrained
        # model setup
        self.model_name = model_name
        # training hyper parameter
        self.image_count, self.batch_size, self.num_worker, self.lr, self.step_size, self.gamma, self.momentum, self.num_epochs = \
            image_count, batch_size, num_worker, lr, step_size, gamma, momentum, num_epochs

        self.out_path = self.update_out_path()

        model_path = self.out_path + 'model.pth'
        if resume and os.path.isfile(model_path) and os.path.isfile(self.out_path + 'metrics.json'):
            self.checkpoint = torch.load(model_path, map_location=device)
        else:
            if resume:
                raise AssertionError(f'no pretrained model or metrics found at {model_path}\n please train model first')
            self.checkpoint = None
        self.full_ds = get_datasets(base_scene, raw_trains, 10000, y_val, resize=resize, X_val=X_val)
        train_size, val_size = int(0.8 * self.image_count), int(0.2 * self.image_count)
        self.ds = {
            'train': Subset(self.full_ds, arange(train_size)),
            'val': Subset(self.full_ds, arange(train_size, train_size + val_size))
        }
        self.dl = {
            'train': DataLoader(self.ds['train'], batch_size=self.batch_size, num_workers=self.num_worker),
            'val': DataLoader(self.ds['val'], batch_size=self.batch_size, num_workers=self.num_worker)
        }
        ##########################
        # ['west', 'east'] = 2
        # ['none'] + color + length + walls + roofs + wheel_count + load_obj = 22
        # dim out really label classes len?
        ##########################
        dim_out = self.full_ds.dim_out
        if loss == 'MSELoss':
            loss_fn = nn.MSELoss()
        else:
            loss_fn = nn.CrossEntropyLoss()
        self.criteria = [loss_fn] * dim_out
        self.model = get_model(model_name, pretrained, dim_out)

        if self.checkpoint is not None:
            self.model.load_state_dict(self.checkpoint['model_state_dict'])

        if optimizer_ == 'SGD':
            optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=momentum)
        elif optimizer_ == 'ADAM':
            optimizer = optim.Adam(self.model.parameters(), lr=lr)
        else:
            raise AssertionError('specify valid optimizer')
        if self.checkpoint is not None:
            optimizer.load_state_dict(self.checkpoint['optimizer_state_dict'])
        self.optimizer = optimizer

        # apparently the optimizer is not on gpu -> send it to the gpu
        optimizer_to(self.optimizer, device=device)

        self.scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        # self.scheduler = lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)

    def cross_val_train(self, train_size, n_splits=5, ):

        random_state = 0
        test_size = 2000
        y = np.concatenate([self.full_ds.get_direction(item) for item in range(self.full_ds.__len__())])
        for training_size in train_size:
            self.image_count = training_size
            cv = StratifiedShuffleSplit(train_size=training_size, test_size=test_size, random_state=random_state,
                                        n_splits=n_splits)
            for fold, (tr_idx, val_idx) in enumerate(cv.split(np.zeros(len(y)), y)):
                self.out_path = self.update_out_path(prefix=f'cv/', suffix=f'it_{fold}/')
                self.ds = {
                    'train': Subset(self.full_ds, tr_idx),
                    'val': Subset(self.full_ds, val_idx)
                }
                self.dl = {
                    'train': DataLoader(self.ds['train'], batch_size=self.batch_size, shuffle=True,
                                        num_workers=self.num_worker),
                    'val': DataLoader(self.ds['val'], batch_size=self.batch_size, shuffle=True,
                                      num_workers=self.num_worker)
                }

                self.model = do_train_reg(self.base_scene, self.raw_trains, self.y_val, self.device, self.out_path,
                                          self.model_name, self.model, self.full_ds, self.dl, self.checkpoint,
                                          self.optimizer, self.scheduler, self.criteria,
                                          num_epochs=self.num_epochs, lr=self.lr, step_size=self.step_size,
                                          gamma=self.gamma,
                                          )
                torch.cuda.empty_cache()

    def train(self):
        self.model = do_train_reg(self.base_scene, self.raw_trains, self.y_val, self.device, self.out_path,
                                  self.model_name,
                                  self.model, self.full_ds, self.dl, self.checkpoint, self.optimizer, self.scheduler,
                                  self.criteria, num_epochs=self.num_epochs, lr=self.lr, step_size=self.step_size,
                                  gamma=self.gamma,
                                  )
        torch.cuda.empty_cache()

    def predict_world_coords(self):
        ds = self.full_ds
        rtpt = RTPT(name_initials='LH', experiment_name=f'Pred_{self.base_scene[:3]}_{self.raw_trains[0]}',
                    max_iterations=ds.__len__())
        rtpt.start()
        all_preds = []
        self.model.to(self.device)
        with torch.no_grad():
            for item in range(ds.__len__()):
                item_preds = np.zeros((ds.output_dim, ds.class_dim))
                masks = ds.get_pred_masks(item)
                image = ds.get_pil_image(item)
                image = ds.norm(image)
                masks, image = masks.to(self.device), image.to(self.device)
                for id, mask in enumerate(masks):
                    mask = mask.unsqueeze(dim=0) / len(ds.attribute_classes)
                    inputs = torch.concat([image, mask])
                    inputs = inputs.unsqueeze(dim=0)
                    outputs = self.model(inputs)
                    preds = outputs.to("cpu").detach().numpy()
                    item_preds[id, :] = preds
                all_preds.append(item_preds)
                rtpt.step()
        all_preds = np.array(all_preds)

        path = f'output/predictions/{self.raw_trains}/{self.base_scene}'
        os.makedirs(path, exist_ok=True)
        np.save(path + f'/world_coord_predictions.npy', all_preds, allow_pickle=True)
        # with open(path + f'/world_coord_predictions.json', 'w+') as fp:
        #     json.dump(all_preds, fp, indent=2)

    def update_out_path(self, prefix='', suffix=''):
        pre = '_pretrained' if self.pretrained else ''
        config = f'imcount_{self.image_count}_X_val_{self.X_val}{pre}_lr_{self.lr}_step_{self.step_size}_gamma{self.gamma}'
        out_path = f'output/models/{self.model_name}/{self.y_val}_classification/{self.raw_trains}' \
                   f'/{self.base_scene}/{prefix}{config}/{suffix}'
        return out_path

    def plot_loss(self):
        with open(self.out_path + 'metrics.json', 'r') as fp:
            statistics = json.load(fp)
        epoch_loss = statistics['epoch_loss']
        epoch_error = statistics['epoch_error']
        path = self.out_path + 'statistics/'
        os.makedirs(path, exist_ok=True)
        plt.ylim(0, 2)
        for phase in ['train', 'val']:
            plt.plot(np.arange(1, self.num_epochs + 1), epoch_loss[phase],
                     label=f'{phase} MSE loss = {round(epoch_loss[phase][-1], 4)}')
        plt.xlabel("training epoch")
        plt.ylabel("loss")
        plt.legend(loc="upper right")
        plt.title(f'{self.model_name} MSE loss')
        plt.savefig(path + f'loss.png', dpi=400)
        plt.close()

        for phase in ['train', 'val']:
            for type in ['x', 'y', 'z', 'total']:
                plt.plot(np.arange(1, self.num_epochs + 1), epoch_error[phase][type],
                         label=f'{type} MAE = {round(epoch_error[phase][type][-1], 4)}')
            plt.xlabel("training epoch")
            plt.ylabel(f"mean absolut error")
            plt.legend(loc="upper right")
            plt.title(f'{self.model_name} {phase} mean absolut error')
            plt.savefig(path + f'{phase}_error.png', dpi=400)
            plt.close()


def do_train_reg(base_scene, raw_trains, y_val, device, out_path, model_name, model, full_ds, dl,
                 checkpoint, optimizer, scheduler, criteria, num_epochs=25, lr=0.001, step_size=5, gamma=.8
                 ):
    rtpt = RTPT(name_initials='LH', experiment_name=f'train_{base_scene[:3]}_{raw_trains[0]}',
                max_iterations=num_epochs)
    rtpt.start()
    epoch_init = 0
    if checkpoint is not None:
        epoch_init = checkpoint['epoch']

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = math.inf

    epoch_error = {
        'train': {},
        'val': {}
    }
    for coord in ['x', 'y', 'z', 'total']:
        epoch_error['train'][coord], epoch_error['val'][coord] = [0] * num_epochs, [0] * num_epochs

    epoch_loss = {
        'train': [0] * num_epochs,
        'val': [0] * num_epochs
    }

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
            all_labels = np.empty((3, 0))
            all_preds = np.empty((3, 0))
            losses = np.empty((3, 0))

            # Iterate over data.
            for inputs, labels in dl[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                model.to(device)
                labels = torch.t(labels)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    outputs = torch.t(outputs)
                    preds = outputs
                    # plot_masked_im(inputs[0], labels[:, 0])

                    losses = [criterion(output, label) for label, output, criterion in zip(labels, outputs, criteria)]
                    loss = sum(losses)
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # to numpy
                labels, preds = labels.to("cpu"), preds.to("cpu")
                labels, preds = labels.detach().numpy(), preds.detach().numpy()
                all_labels = np.hstack((all_labels, labels))
                all_preds = np.hstack((all_preds, preds))
            if phase == 'train':
                scheduler.step()
            epoch_error[phase]['x'][epoch] = mean_absolute_error(all_labels[0], all_preds[0])
            epoch_error[phase]['y'][epoch] = mean_absolute_error(all_labels[1], all_preds[1])
            epoch_error[phase]['z'][epoch] = mean_absolute_error(all_labels[2], all_preds[2])
            epoch_error[phase]['total'][epoch] = mean_absolute_error(all_labels, all_preds)
            epoch_loss[phase][epoch] = mean_squared_error(all_labels, all_preds)

            print('{} Loss: {:.4f}, mean absolut error: {:.4f}'.format(phase, epoch_loss[phase][epoch],
                                                                       epoch_error[phase]['total'][epoch]))

            # deep copy the model
            if phase == 'val' and epoch_loss[phase][epoch] < best_loss:
                best_loss = epoch_loss[phase][epoch]
                best_model_wts = copy.deepcopy(model.state_dict())

    print('Best val Loss: {:4f}'.format(best_loss))
    os.makedirs(out_path, exist_ok=True)

    # load best model weights
    model.load_state_dict(best_model_wts)
    statistics = {
        'epoch_error': epoch_error,
        'epoch_loss': epoch_loss
    }

    torch.save({
        'epoch': num_epochs + epoch_init,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss
    }, out_path + 'model.pth')

    if checkpoint is not None:
        with open(out_path + 'metrics.json', 'r') as fp:
            stat_init = json.load(fp)
            statistics = merge(stat_init.copy(), statistics)
    with open(out_path + 'metrics.json', 'w+') as fp:
        json.dump(statistics, fp)
    return model.to('cpu')
