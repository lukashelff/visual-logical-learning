import pandas as pd
from rtpt.rtpt import RTPT
from sklearn.metrics import accuracy_score
from tabulate import tabulate
from torch.utils.data import DataLoader

from michalski_trains.dataset import get_datasets
from util import *


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