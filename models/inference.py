import pandas as pd
from rtpt.rtpt import RTPT
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from torch.utils.data import DataLoader

from michalski_trains.dataset import get_datasets
from util import *


def infer_symbolic(trainer, use_transfer_trained_model=False, im_counts=None):
    # model_pred1 = 'attr_predictor'
    # model_pred2 = 'resnet18'
    # out_path = f'output/models/{model_pred2}/attribute_classification/RandomTrains/{trainer.base_scene}'
    # config = f'imcount_10000_X_val_predicted_mask_lr_0.001_step_5_gamma0.8'
    raw_trains = trainer.raw_trains
    dl = DataLoader(trainer.full_ds, batch_size=trainer.batch_size, shuffle=False, num_workers=4)
    if im_counts is None: im_counts = [100, 1000, 8000]
    if not use_transfer_trained_model:
        trainer.raw_trains = 'RandomTrains'
        im_counts = [8000]

    # trainer.X_val = 'gt_mask'

    for im_count in im_counts:
        out_path = f'output/models/inferred_symbolic/{trainer.model_name}/attribute_classification/{trainer.setting}'

        print(
            f'{trainer.model_name} trained on {im_count}{trainer.raw_trains} images predicting train descriptions for the'
            f' {raw_trains} trains in {trainer.base_scene}')
        accs = []
        for fold in range(5):

            path = trainer.get_model_path(prefix=f'cv/', suffix=f'it_{fold}/')
            trainer.setup_model(path=path, resume=True)

            trainer.model.eval()  # Set model to evaluate mode
            trainer.model.to(trainer.device)

            rtpt = RTPT(name_initials='LH', experiment_name=f'infer_symbolic_{trainer.base_scene[:3]}',
                        max_iterations=trainer.full_ds.__len__() / trainer.batch_size)
            rtpt.start()

            with torch.no_grad():

                all_labels = np.empty([0, 32], dtype=int)
                all_preds = np.empty([0, 32], dtype=int)

                # Iterate over data.
                for inputs, labels in dl:
                    # for item in range(trainer.full_ds.__len__()):

                    inputs = inputs.to(trainer.device)
                    labels = labels.to(trainer.device)
                    trainer.model.to(trainer.device)

                    outputs = trainer.model(inputs)
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
            del trainer.model
        print('average acc score: ' + str(np.mean(accs)))


def transfer_classification(trainer, train_size, n_splits=5, batch_size=None):
    print(f'transfer classification: {trainer.model_name} trained on base scene to predict other scenes')
    data = pd.DataFrame(columns=['methods', 'number of images', 'scenes', 'mean', 'variance', 'std'])
    data_cv = pd.DataFrame(columns=['Methods', 'number of images', 'cv iteration', 'Validation acc', 'scene'])
    train_size = [100, 1000, 8000] if train_size is None else train_size
    batch_size = trainer.batch_size if batch_size is None else batch_size

    rtpt = RTPT(name_initials='LH', experiment_name=f'trans', max_iterations=n_splits * 4 * len(train_size))
    rtpt.start()

    for scene in ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']:
        ds = get_datasets(scene, trainer.raw_trains, trainer.train_vis, 10000, resize=False,
                          ds_path=trainer.ds_path)
        dl = DataLoader(ds, batch_size=batch_size, num_workers=trainer.num_worker)
        for training_size in train_size:
            accs = []
            for fold in range(n_splits):
                rtpt.step()
                torch.cuda.memory_summary(device=None, abbreviated=False)

                trainer.out_path = trainer.get_model_path(prefix=f'cv/', suffix=f'it_{fold}/')
                del trainer.model
                trainer.setup_model(resume=True)
                trainer.model.eval()

                all_labels = np.empty(0, int)
                all_preds = np.empty(0, int)
                # Iterate over data.
                for inputs, labels in dl:
                    print(torch.cuda.memory_summary(device=None, abbreviated=False))
                    inputs = inputs.to(trainer.device)
                    labels = labels.to(trainer.device)
                    trainer.model.to(trainer.device)
                    labels = torch.t(labels)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    print(torch.cuda.memory_summary(device=None, abbreviated=False))

                    outputs = trainer.model(inputs)
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

                print(f'{trainer.model_name} trained on base scene with {training_size} images (cv iteration {fold})'
                      f' achieves an accuracy of {acc} when classifying {scene} images')
                li = [trainer.model_name, training_size, fold, acc, scene]
                _df = pd.DataFrame([li], columns=['Methods', 'number of images', 'cv iteration', 'Validation acc',
                                                  'scene'])
                data_cv = pd.concat([data_cv, _df], ignore_index=True)
            mean = sum(accs) / len(accs)
            variance = sum((xi - mean) ** 2 for xi in accs) / len(accs)
            std = np.sqrt(variance)
            li = [trainer.model_name, training_size, scene, mean, variance, std]
            _df = pd.DataFrame([li], columns=['methods', 'number of images', 'scenes', 'mean', 'variance', 'std'])
            data = pd.concat([data, _df], ignore_index=True)
    print(tabulate(data, headers='keys', tablefmt='psql'))
    path = f'output/models/{trainer.model_name}/{trainer.y_val}_classification/{trainer.raw_trains}/{trainer.base_scene}/'
    os.makedirs(path, exist_ok=True)
    data.to_csv(path + 'transfer_classification.csv')
    data_cv.to_csv(path + 'transfer_classification_cv.csv')
    # csv_to_tex_table(path + 'transfer_classification.csv', )
