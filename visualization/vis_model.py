import json

import matplotlib.colors as mcolors
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from torch.utils.data import random_split

from util import *


def visualize_statistics(raw_trains, base_scene, y_val, datasets, out_path, model_name):
    print('visualizing model statistics from metrics.json')
    # full_ds = get_datasets(base_scene, raw_trains, image_count, y_val, X_val=X_val)
    # train_size, val_size = int(0.7 * image_count), int(0.3 * image_count)
    # train_dataset, val_dataset = random_split(full_ds, [train_size, val_size])
    # datasets = {
    #     'train': train_dataset,
    #     'val': val_dataset
    # }
    if y_val == 'direction':
        vis_statistics_dir(raw_trains, base_scene, model_name, datasets, out_path)
    elif y_val == 'attribute':
        vis_statistics_at(raw_trains, base_scene, model_name, datasets, out_path)


def vis_statistics_dir(raw_trains, base_scene, model_name, datasets, out_path):
    with open(out_path + 'metrics.json', 'r') as fp:
        statistics = json.load(fp)
    label_names = datasets['train'].dataset.labels
    label_classes = datasets['train'].dataset.label_classes
    epoch_label_accs = statistics['epoch_label_accs']
    epoch_loss = statistics['epoch_loss']
    num_epochs = len(epoch_loss['train'])

    path = out_path + 'statistics/'
    os.makedirs(path, exist_ok=True)
    baselines_n = {
        'bal_acc': 'rand choice',
        'acc': 'label freq'
    }
    baselines = get_baselines(datasets)
    unique_label_names = list(set(label_names))
    label_name = unique_label_names[0]
    colors = list(mcolors.TABLEAU_COLORS.values())
    color_ind = 0

    # plot accuracy and balanced accuracy for training and validation
    for phase in ['train', 'val']:
        for acc_type in ['bal_acc', 'acc']:
            plt.plot(np.arange(1, num_epochs + 1), [baselines[phase][acc_type][label_name]] * num_epochs,
                     dash_capstyle='round', color=colors[color_ind], linestyle='dashed')
            final_acc = np.round(epoch_label_accs[phase][label_name][acc_type][-1] * 100)
            plt.plot(np.arange(1, num_epochs + 1), epoch_label_accs[phase][label_name][acc_type],
                     label=f'{phase} {acc_type} = {int(final_acc)}%', color=colors[color_ind])
            color_ind += 1
    plt.xlabel("training epoch")
    plt.ylabel("acc")
    plt.legend(loc="lower right")
    plt.title(f'{model_name} trained on {base_scene} DS with {raw_trains}')
    plt.savefig(path + f'accuracy.png', dpi=400)
    plt.close()

    # plot loss
    for phase, color in zip(['train', 'val'], colors[:2]):
        plt.plot(np.arange(1, num_epochs + 1), epoch_loss[phase], label=f'{phase} loss', color=color)
    plt.xlabel("training epoch")
    plt.ylabel("acc")
    plt.ylim(ymin=0)
    plt.legend(loc="lower right")
    plt.title(f'{model_name} loss on {base_scene} with {raw_trains} DS')
    plt.savefig(path + f'loss.png', dpi=400)
    plt.close()

    # plot precision and recall
    for metric in ['precision', 'recall']:
        color_ind = 0
        for phase in ['train', 'val']:
            for class_name in label_classes:
                class_id = label_classes.index(class_name)
                final_acc = np.round(epoch_label_accs[phase][metric][class_name][-1] * 100)
                plt.plot(np.arange(1, num_epochs + 1), epoch_label_accs[phase][metric][class_name],
                         label=f'{class_name} {phase} = {int(final_acc)}%', color=colors[color_ind])
                color_ind += 1
        plt.xlabel("training epoch")
        plt.ylim(ymin=0)
        plt.ylabel(metric)
        plt.legend(loc="lower right")
        plt.title(f'{metric} of {label_name} label during {phase}\n {base_scene} DS with {raw_trains}')
        plt.savefig(path + f'{metric}_performance_{label_name}.png', dpi=400)
        plt.close()
    print(f'statistics saved in {path}')


def vis_statistics_at(raw_trains, base_scene, model_name, datasets, out_path):
    with open(out_path + 'metrics.json', 'r') as fp:
        statistics = json.load(fp)
    attributes = datasets['train'].dataset.attributes
    classes_per_attribute = datasets['train'].dataset.classes_per_attribute

    epoch_label_accs = statistics['epoch_label_accs']
    epoch_acum_accs = statistics['epoch_acum_accs']
    epoch_loss = statistics['epoch_loss']
    num_epochs = len(epoch_loss['train'])
    path = out_path + 'statistics/'
    os.makedirs(path, exist_ok=True)
    colors = list(mcolors.TABLEAU_COLORS.values())

    # get baselines
    baselines_n = {
        'bal_acc': 'rand choice',
        'acc': 'label freq'
    }
    baselines = get_baselines(datasets)

    # combine load metrics
    for phase in ['train', 'val']:
        epoch_label_accs[phase]['load_obj'] = {}
        for metric_type in ['acc', 'bal_acc']:
            all_d = [epoch_label_accs[phase]['load_1'][metric_type], epoch_label_accs[phase]['load_2'][metric_type],
                     epoch_label_accs[phase]['load_3'][metric_type]]
            epoch_label_accs[phase]['load_obj'][metric_type] = np.mean(all_d, axis=0)

    # plot train/validation accuracy/balanced accuracy for every label
    for phase in ['train', 'val']:
        for acc_type in ['bal_acc', 'acc']:
            for label_name, color in zip(attributes, colors[:len(attributes)]):
                plt.plot(np.arange(1, num_epochs + 1), [baselines[phase][acc_type][label_name]] * num_epochs,
                         dash_capstyle='round', color=color, linestyle='dashed')
                final_acc = np.round(epoch_label_accs[phase][label_name][acc_type][-1] * 100)
                plt.plot(np.arange(1, num_epochs + 1), epoch_label_accs[phase][label_name][acc_type],
                         label=f'{label_name} {acc_type} = {int(final_acc)}%',
                         color=color)
            plt.xlabel("training epoch")
            plt.ylabel("acc")
            plt.ylim(ymin=0)
            plt.legend(loc="lower right")
            plt.title(f'{phase} {acc_type} for each label including {baselines_n[acc_type]} as baseline\n'
                      f'model trained on {base_scene} DS with {raw_trains}')
            plt.savefig(path + f'{phase}_{acc_type}.png', dpi=400)
            plt.close()

    # plot accumulated train/validation accuracy and balanced accuracy
    for phase, acc_type, color in [['train', 'bal_acc', colors[0]], ['train', 'acc', colors[1]],
                                   ['val', 'bal_acc', colors[2]], ['val', 'acc', colors[3]]]:
        final_acc = np.round(epoch_acum_accs[phase][acc_type][-1] * 100)
        plt.ylim(ymin=0)
        plt.plot(np.arange(1, num_epochs + 1), epoch_acum_accs[phase][acc_type], color=color,
                 label=f'{phase} {acc_type} = {int(final_acc)}%')
        plt.plot(np.arange(1, num_epochs + 1), [np.mean(list(baselines[phase][acc_type].values()))] * num_epochs,
                 dash_capstyle='round', color=color, linestyle='dashed')
    plt.xlabel("training epoch")
    plt.ylabel("acc")
    plt.legend(loc="lower right")
    plt.title(f'Accumulated classification accuracy including corresponding baseline\n'
              f'{model_name} model trained on {base_scene} DS with {raw_trains}')
    plt.savefig(path + f'accumulated_acc.png', dpi=400)
    plt.close()

    # plot loss
    for phase, color in zip(['train', 'val'], colors[:2]):
        plt.plot(np.arange(1, num_epochs + 1), epoch_loss[phase], label=f'{phase} loss', color=color)
    plt.xlabel("training epoch")
    plt.ylabel("acc")
    plt.legend(loc="lower right")
    plt.title(f'{model_name} loss on {base_scene} with {raw_trains} DS')
    plt.savefig(path + f'loss.png', dpi=400)
    plt.close()

    # plot precision and recall for every class label
    path = path + 'label_performance_measure/'
    os.makedirs(path, exist_ok=True)
    for phase in ['train', 'val']:
        for metric in ['precision', 'recall']:
            for label_classes, label_name in zip(classes_per_attribute, attributes):
                color_ind = 0
                for class_name in label_classes:
                    final_acc = np.round(epoch_label_accs[phase][metric][class_name][-1] * 100)
                    plt.plot(np.arange(1, num_epochs + 1), epoch_label_accs[phase][metric][class_name],
                             label=f'{class_name} {metric} = {int(final_acc)}%', color=colors[color_ind])
                    color_ind += 1
                plt.xlabel("training epoch")
                plt.ylim(ymin=0)
                plt.ylabel(metric)
                plt.legend(loc="lower right")
                plt.title(f'{metric} of {label_name} label during {phase}\n {base_scene} DS with {raw_trains}')
                plt.savefig(path + f'{metric}_performance_{phase}_{label_name}.png', dpi=400)
                plt.close()


def visualize_model(model, class_names, device, dataloaders, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images // 2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                plt.imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)


def vis_confusion_matrix(raw_trains, base_scene, out_path, model_name, model, dl, device):
    print('evaluate model validation performance and create confusion matrix')
    num_epoch = 1
    # model, _, _, _, dl, ds_val, _, _ = train_setup(backbone, model_name, base_scene, raw_trains,
    #                                            y_val, device,
    #                                            num_epoch, out_path, resume=True, batch_size=20,
    #                                            image_count=image_count, )

    model.eval()  # Set model to evaluate mode

    all_labels = np.empty(0, int)
    all_preds = np.empty(0, int)
    all_trains = []

    # Iterate over data.
    for inputs, labels in dl['val']:

        inputs = inputs.to(device)
        labels = labels.to(device)
        model.to(device)
        labels = torch.t(labels)

        outputs = model(inputs)
        if outputs.dim() < 3:
            outputs = outputs.unsqueeze(dim=1)
        outputs = torch.moveaxis(outputs, 0, 1)

        preds = torch.max(outputs, dim=2)[1]

        labels, preds = labels.to("cpu"), preds.to("cpu")
        labels, preds = labels.detach().numpy(), preds.detach().numpy()
        all_labels = np.hstack((all_labels, labels.flatten()))
        all_preds = np.hstack((all_preds, preds.flatten()))
    c_matrix = confusion_matrix(all_labels, all_preds, normalize='all')
    acc = accuracy_score(all_labels, all_preds)
    ConfusionMatrixDisplay.from_predictions(all_labels, all_preds, include_values=False, cmap=plt.cm.Blues,
                                            normalize='true')
    plt.title(f'Confusion Matrix for {model_name} (acc: {round(acc * 100, 2)}%)\n'
              f'trained on {raw_trains} in {base_scene} DS')
    pth = out_path + 'statistics'
    os.makedirs(pth, exist_ok=True)
    plt.savefig(pth + '/confusion_matrix', dpi=400)
    plt.show()
    plt.close()
    # calculate confusion matrix
