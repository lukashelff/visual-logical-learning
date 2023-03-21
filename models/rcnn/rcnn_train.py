import copy
import time
from rtpt.rtpt import RTPT
from tqdm import tqdm

from models.rcnn.engine import train_one_epoch, evaluate
from util import *


def train_rcnn(base_scene, train_col, y_val, device, out_path, model_name, model, full_ds, dl,
               checkpoint, optimizer, scheduler, criteria, num_epochs=25, lr=0.001, step_size=5, gamma=.8,
               save_model=True, rtpt_extra=0, train='train'):
    rtpt = RTPT(name_initials='LH', experiment_name=f'train_{base_scene[:3]}_{train_col[0]}',
                max_iterations=num_epochs + rtpt_extra)
    rtpt.start()
    train_loss_hist = Averager()
    val_loss_hist = Averager()
    train_itr = 1
    val_itr = 1
    # train and validation loss lists to store loss values of all...
    # ... iterations till ena and plot graphs for all iterations
    train_loss_list = []
    val_loss_list = []

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

    model.to(device)

    for epoch in range(num_epochs):
        rtpt.step()

        print(f"EPOCH {epoch + 1} of {num_epochs}")

        # reset the training and validation loss histories for the current epoch
        train_loss_hist.reset()
        val_loss_hist.reset()

        # start timer and carry out training and validation
        start = time.time()
        # train for one epoch, printing every 10 iterations
        train_one_epoch(model, optimizer, dl['train'], device, scheduler, epoch, print_freq=100)
        # update the learning rate
        scheduler.step()
        # evaluate on the test dataset
        evaluate(model, dl['val'], device=device)

        # train_loss, train_itr = do_train(dl['train'], model, optimizer, device, train_loss_hist, scheduler)
        # val_loss, val_itr = validate(dl['val'], model, device, val_loss_hist)
        # print(f"Epoch #{epoch + 1} train loss: {train_loss_hist.value:.3f}")
        # print(f"Epoch #{epoch + 1} validation loss: {val_loss_hist.value:.3f}")
        end = time.time()
        print(f"Took {((end - start) / 60):.3f} minutes for epoch {epoch}")

    # plot_prediction(model, dl['val'], device)
    os.makedirs(out_path, exist_ok=True)
    if save_model:
        torch.save({
            'epoch': num_epochs + epoch_init,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': val_loss
        }, out_path + 'model.pth')


# function for running training iterations
def do_train(train_data_loader, model, optimizer, device, train_loss_hist, scheduler):
    print('Training')

    # initialize tqdm progress bar
    prog_bar = tqdm(train_data_loader, total=len(train_data_loader))
    train_loss_list = []
    train_itr = 0

    for i, data in enumerate(prog_bar):
        optimizer.zero_grad()
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        train_loss_list.append(loss_value)

        train_loss_hist.send(loss_value)

        losses.backward()
        optimizer.step()

        train_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    scheduler.step()
    return train_loss_list, train_itr


# function for running validation iterations
def validate(valid_data_loader, model, device, val_loss_hist):
    print('Validating')
    val_loss_list = []
    val_itr = 0

    # initialize tqdm progress bar
    prog_bar = tqdm(valid_data_loader, total=len(valid_data_loader))

    # plot_prediction(model, valid_data_loader, device)

    for i, data in enumerate(prog_bar):
        images, targets = data

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with torch.no_grad():
            loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())
        loss_value = losses.item()
        val_loss_list.append(loss_value)
        val_loss_hist.send(loss_value)
        val_itr += 1

        # update the loss value beside the progress bar for each iteration
        prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")
    return val_loss_list, val_itr


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
