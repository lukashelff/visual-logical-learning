import copy
import json
import time

from rtpt.rtpt import RTPT
from sklearn.metrics import balanced_accuracy_score, accuracy_score, confusion_matrix

from util import *


def do_train(base_scene, raw_trains, y_val, device, out_path, model_name, model, full_ds, dl,
             checkpoint, optimizer, scheduler, criteria, num_epochs=25, lr=0.001, step_size=5, gamma=.8,
             save_model=True, rtpt_extra=0, train='train', ex_name=None):
    ex_name = f'train_{base_scene[:3]}_{raw_trains[0]}' if ex_name is None else ex_name
    rtpt = RTPT(name_initials='LH', experiment_name=ex_name,
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
                        rtpt.step()

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
