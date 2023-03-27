import glob
import json

import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier, RidgeClassifier
from sklearn.metrics import accuracy_score
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, train_test_split, KFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from torch.utils.data import random_split
import wittgenstein as lw
import seaborn as sns
from m_train_dataset import get_datasets
from models.trainer import Trainer
from util import *


def mean_variance_comparison(full_ds, train_size):
    random_state = 0
    test_size = 2000
    n_splits = 10
    scores = []
    scores_mean = []
    classifiers = {
        'RandomForestClassifier': RandomForestClassifier(max_depth=2, random_state=0),
        'KNeighborsClassifier': KNeighborsClassifier(n_neighbors=3),
        'ExtraTreeClassifier': ExtraTreeClassifier(random_state=0),
        'DecisionTreeClassifier': DecisionTreeClassifier(random_state=0),
        'Multi-Layer Perceptron NN': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, 200),
                                                   random_state=random_state,
                                                   max_iter=10000),
        'RidgeClassifier': RidgeClassifier(random_state=0),
        # not applicable
        # 'Gaussian Naive Bayes': GaussianNB(),
        # 'Stochastic Gradient Descent': SGDClassifier(random_state=random_state),
        # 'Support Vector Machine': svm.SVC(random_state=random_state),
        # 'Ada Boost': AdaBoostClassifier(
        #     base_estimator=RandomForestClassifier(n_estimators=500, min_samples_split=2, random_state=random_state),
        #     random_state=random_state, n_estimators=100, learning_rate=0.1),
        # 'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, learning_rate=1, max_depth=3,
        #                                                 random_state=random_state),
        # applicable
        # 'K-Neighbors': KNeighborsClassifier(n_neighbors=10, weights="distance"),

        # 'Histogram Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
        # 'Irep': lw.IREP(random_state=random_state),
        # 'Ripper': lw.RIPPER(random_state=random_state),
    }
    # learning relational rules,
    # Examples or target concept may require relational representation that includes multiple entities with relationships between them
    # As you know, first-order logic is a more powerful representation for handling such relational descriptions
    # Consistency, Completeness
    # pruning?
    # print(f'supervised learning methods on michalski train attributes to predict the corresponding labels')
    # print(f'the methods described below were employed on a michalski train dataset of the sizes {image_counts}')
    # print(f'the score represents the methods mean accuracy over {n_splits} folds of cross-validation')
    print('extract ydir')
    y_dir = np.concatenate([full_ds.get_direction(item) for item in range(full_ds.__len__())])
    print('extract X')
    X = np.array([x.flatten().numpy() for x, _ in full_ds])
    print('extract y')
    y = np.array([y.numpy() for _, y in full_ds])

    for training_size in train_size:
        print(f'employing methods on dataset with a size of {training_size} images')
        # normalize input data
        # scaler = StandardScaler()
        # scaler.fit(X)
        # X = scaler.transform(X)
        cv = StratifiedShuffleSplit(train_size=training_size, test_size=test_size, random_state=random_state,
                                    n_splits=n_splits)

        for clf_name, clf in classifiers.items():
            print(f'employing {clf_name}')
            mean_score = 0
            for cv_it, (train_index, test_index) in enumerate(cv.split(np.zeros(len(y)), y_dir)):
                X_train, X_test = X[train_index], X[test_index]
                y_train, y_test = y[train_index], y[test_index]
                clf.fit(X_train, y_train)
                y_pred = clf.predict(X_test)
                y_pred = y_pred.reshape((y_pred.shape[0] * 32, 22))
                y_test = y_test.reshape((y_test.shape[0] * 32, 22))
                # [np.where(pred == 1) for pred in y_pred.reshape((y_pred.shape[0] * 32, 22))]
                acc_score = accuracy_score(y_test, y_pred)
                scores.append([clf_name, str(training_size) + ' images', cv_it, acc_score])
                mean_score += acc_score / n_splits
            print(mean_score)
            scores_mean.append([clf_name, str(training_size) + ' images', mean_score])
    df_cross_val = pd.DataFrame(scores, columns=['classification methods', 'number of images', 'cv iteration', 'score'])

    sns.set_theme(style="whitegrid")
    f, ax = plt.subplots()
    # Show each observation with a scatterplot
    sns.stripplot(x='score', y='classification methods', hue='number of images', data=df_cross_val,
                  dodge=True,
                  alpha=.25,
                  zorder=1,
                  jitter=False
                  )

    # Show the conditional means, aligning each pointplot in the
    # center of the strips by adjusting the width allotted to each
    # category (.8 by default) by the number of hue levels
    sns.pointplot(x='score', y='classification methods', hue='number of images', data=df_cross_val,
                  dodge=False,
                  join=False,
                  # palette="dark",
                  markers="d",
                  scale=.75,
                  ci=None
                  )

    plt.title('Comparison of methods learning composition of michalski train using world cords and label')
    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    length = len(handles) // 2
    # length = 0

    ax.legend(handles[length:], labels[length:], title="number of images",
              handletextpad=0,
              loc="upper right", frameon=True)
    path = 'output/models/predict_composition/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'supervised_lr_mean_variance_comparison.png', bbox_inches='tight', dpi=400)
    plt.close()


def compare_multi_class(base_scene, raw_trains, y_val='attribute_str', X_val='gt_positions'):
    print('Multiclass and multi-output algorithms')
    full_ds = get_datasets(base_scene, raw_trains, 10000, y_val=y_val, X_val=X_val)
    X = np.concatenate([x.flatten().unsqueeze(dim=0) for x, y in full_ds], axis=0)
    # y = np.concatenate([y.unsqueeze(dim=0) for x, y in full_ds], axis=0)
    y = np.array([y for x, y in full_ds])
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    clf = RandomForestClassifier(max_depth=2, random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = sum(sum(pred == y_test)) / pred.size
    print('RandomForestClassifier acc ' + str(acc))

    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = sum(sum(pred == y_test)) / pred.size
    print('KNeighborsClassifier (neighbor=3) acc ' + str(acc))

    clf = ExtraTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = sum(sum(pred == y_test)) / pred.size
    print('ExtraTreeClassifier  acc ' + str(acc))

    clf = DecisionTreeClassifier(random_state=0)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    acc = sum(sum(pred == y_test)) / pred.size
    print('ExtraTreeClassifier  acc ' + str(acc))

    print('methods using binary labels (single class multi output)')


def predict_composition(full_ds):
    random_state = 0
    test_size = 2000
    train_size = 8000
    n_splits = 5
    scores = []
    scores_mean = []
    classifiers = {
        'Multi-Layer Perceptron NN': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, 200),
                                                   random_state=random_state, max_iter=1000),
    }
    # learning relational rules,
    # Examples or target concept may require relational representation that includes multiple entities with relationships between them
    # As you know, first-order logic is a more powerful representation for handling such relational descriptions
    # Consistency, Completeness
    # pruning?
    # print(f'supervised learning methods on michalski train attributes to predict the corresponding labels')
    # print(f'the methods described below were employed on a michalski train dataset of the sizes {image_counts}')
    # print(f'the score represents the methods mean accuracy over {n_splits} folds of cross-validation')

    print('extract X')
    X = np.array([x.flatten().numpy() for x, _ in full_ds])
    print('extract y')
    y = np.array([y.numpy() for _, y in full_ds])
    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, 200), random_state=random_state,
                        max_iter=1000)
    clf.fit(X,y)
    score = clf.score(X, y)
    # scores = cross_val_score(clf, X, y, cv=5)
    # acc = scores.mean()
    print(f'multi-layer perceptron score: {score}%')
    print('-' * 10)

    # print(f'prediction michalski train attribute composition using mlp')
    # cv = KFold(n_splits=n_splits)
    # for clf_name, clf in classifiers.items():
    #     print(f'employing {clf_name}')
    #     mean_score = 0
    #     for cv_it, (train_index, test_index) in enumerate(cv.split(y)):
    #         X_train, X_test = X[train_index], X[test_index]
    #         y_train, y_test = y[train_index], y[test_index]
    #         clf.fit(X_train, y_train)
    #         y_pred = clf.predict(X_test)
    #         y_pred = y_pred.reshape((y_pred.shape[0] * 32, 22))
    #         y_test = y_test.reshape((y_test.shape[0] * 32, 22))
    #         # [np.where(pred == 1) for pred in y_pred.reshape((y_pred.shape[0] * 32, 22))]
    #         acc_score = accuracy_score(y_test, y_pred)
    #         scores.append([clf_name, str(train_size) + ' images', cv_it, acc_score])
    #         mean_score += acc_score / n_splits
    #     print(mean_score)
    #     scores_mean.append([clf_name, str(train_size) + ' images', mean_score])
    # df_cross_val = pd.DataFrame(scores, columns=['classification methods', 'number of images', 'cv iteration', 'score'])
    #
    # sns.set_theme(style="whitegrid")
    # f, ax = plt.subplots()
    # # Show each observation with a scatterplot
    # sns.stripplot(x='score', y='classification methods', hue='number of images', data=df_cross_val,
    #               dodge=True,
    #               alpha=.25,
    #               zorder=1,
    #               jitter=False
    #               )
    #
    # # Show the conditional means, aligning each pointplot in the
    # # center of the strips by adjusting the width allotted to each
    # # category (.8 by default) by the number of hue levels
    # sns.pointplot(x='score', y='classification methods', hue='number of images', data=df_cross_val,
    #               dodge=False,
    #               join=False,
    #               # palette="dark",
    #               markers="d",
    #               scale=.75,
    #               ci=None
    #               )
    #
    # plt.title('Comparison of methods learning composition of michalski train using world cords and label')
    # # Improve the legend
    # handles, labels = ax.get_legend_handles_labels()
    # length = len(handles) // 2
    # # length = 0
    #
    # ax.legend(handles[length:], labels[length:], title="number of images",
    #           handletextpad=0,
    #           loc="upper right", frameon=True)
    # path = 'output/models/predict_composition/'
    # os.makedirs(path, exist_ok=True)
    # plt.savefig(path + 'supervised_lr_mean_variance_comparison.png', bbox_inches='tight')
    # plt.close()
