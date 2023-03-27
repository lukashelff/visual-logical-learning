import glob
import json
import shutil
from pathlib import Path

from sklearn.experimental import enable_halving_search_cv
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.cluster import MiniBatchKMeans
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, \
    ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit, HalvingGridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from torch.utils.data import random_split
import wittgenstein as lw
import seaborn as sns


from util import *


def michalski_learner(base_scene, raw_trains, image_counts, device):
    y_val = 'direction'
    X_val = 'gt_attributes'
    X_val = 'predicted_attributes'
    base_scene = 'base_scene'
    raw_trains = 'MichalskiTrains'
    random_state = 0
    test_size = 2000
    n_splits = 10
    scores = []
    scores_mean = []

    def id(x, y):
        c = 0
        for xi, yi in zip(x, y):
            if xi != yi:
                c += 1
        return c

    classifiers = {
        'Gaussian Naive Bayes': GaussianNB(),
        'Stochastic Gradient Descent': SGDClassifier(random_state=random_state),
        'Support Vector Classifier': svm.SVC(random_state=random_state, cache_size=1000, C=10,
                                             decision_function_shape='ovo'),
        'K-Neighbors': KNeighborsClassifier(n_neighbors=10, weights="distance", metric=id),

        'Multi-Layer Perceptron NN': MLPClassifier(alpha=0.05, learning_rate='adaptive', max_iter=1000),
        'Decision Tree': DecisionTreeClassifier(random_state=random_state),
        'Random Forest': RandomForestClassifier(n_estimators=500, min_samples_split=2, random_state=random_state),
        # 'Extra Trees': ExtraTreesClassifier(random_state=random_state),
        'AdaBoost': AdaBoostClassifier(
            base_estimator=RandomForestClassifier(n_estimators=500, min_samples_split=2, random_state=random_state),
            random_state=random_state, n_estimators=100, learning_rate=0.1),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=500, learning_rate=1, max_depth=3,
                                                        random_state=random_state),
        # 'Ripper': lw.RIPPER(random_state=random_state),
        # 'Histogram Gradient Boosting': HistGradientBoostingClassifier(max_iter=100, random_state=random_state),
        # 'Irep': lw.IREP(random_state=random_state),
    }

    # learning relational rules,
    # Examples or target concept may require relational representation that includes multiple entities with relationships between them
    # As you know, first-order logic is a more powerful representation for handling such relational descriptions
    # Consistency, Completeness
    # pruning?
    print(f'supervised learning methods on michalski train attributes to predict the corresponding labels')
    print(f'the methods described below were employed on a michalski train dataset of the sizes {image_counts}')
    print(f'the score represents the methods mean accuracy over {n_splits} folds of cross-validation')

    scores += get_set_transformer_scores(y_val, raw_trains, base_scene)

    from m_train_dataset import get_datasets
    full_ds = get_datasets(base_scene, raw_trains, 10000, y_val=y_val, X_val=X_val)
    for image_count in image_counts:
        print(f'employing methods on dataset with a size of {image_count} images')
        full_ds.predictions_im_count = image_count
        X = np.concatenate([x for x, y in full_ds], axis=0)
        y = np.concatenate([y for x, y in full_ds], axis=0)

        sss = StratifiedShuffleSplit(train_size=image_count, test_size=test_size, random_state=random_state,
                                     n_splits=n_splits)
        # normalize input data
        scaler = StandardScaler()
        scaler.fit(X)
        X = scaler.transform(X)

        # halving grid search to determine best hyper params
        parameter_space = {
            'criterion': ['gini', 'entropy'],
            'splitter': ['best', 'random'],
            'max_depth': [3, 5, 7, 10, None],
            'max_features': ['auto', 'sqrt', 'log2'],
        }

        base_estimator = DecisionTreeClassifier()

        # parameter_space = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5, 10]}
        # base_estimator = RandomForestClassifier(random_state=0)

        # sh = HalvingGridSearchCV(base_estimator, parameter_space, cv=3, random_state=random_state).fit(X, y)
        # # print(sh.best_estimator_)
        # print('best parameters: ' + str(sh.best_params_))
        #
        # print('cross validation score: ' + str(sh.best_score_))

        for clf_name, clf in classifiers.items():
            temp = cross_val_score(clf, X, y, cv=sss, n_jobs=-1)
            for cv_it, score in enumerate(temp):
                scores.append([clf_name, str(image_count) + ' images', cv_it, score])
            scores_mean.append([clf_name, str(image_count) + ' images', temp.mean()])
            if clf_name == 'Decision Tree':
                clf.fit(X, y)
                tree.plot_tree(clf)
                dtree = tree.export_graphviz(clf)
                path = 'output/induction_learner/'
                os.makedirs(path, exist_ok=True)
                with open(path + f'Decision_Tree_graph_{image_count}.json', 'w+') as f:
                    # f.write(dtree)
                    json.dump(dtree, f, indent=2)
                plt.savefig(path + f'Decision_Tree_{image_count}.png', bbox_inches='tight', dpi=400)
                plt.close()

    df_cross_val = pd.DataFrame(scores, columns=['classification methods', 'number of images', 'cv iteration', 'score'])

    # fig = plt.figure()
    # ax = plt.subplot(111)
    # sns.scatterplot(data=df, x='score', y='classification methods', hue='number of images')
    # plt.title('Supervised learning methods')
    # box = ax.get_position()
    # ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    # ax.legend(loc='upper left', bbox_to_anchor=(1, 1.02), title="number of images")
    # path = 'output/induction_learner/'
    # os.makedirs(path, exist_ok=True)
    # plt.savefig(path + 'supervised_lr_methods.png', bbox_inches='tight')
    # plt.close()

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

    # plt.title('Comparison of Supervised learning methods')
    # Improve the legend
    handles, labels = ax.get_legend_handles_labels()
    length = len(handles) // 2
    # length = 0

    ax.set_ylabel("Classification methods")
    ax.set_xlabel("Validation accuracy")

    ax.legend(handles[length:], labels[length:], title="Number of images",
              handletextpad=0,
              loc="lower left", frameon=True)
    path = 'output/induction_learner/'
    os.makedirs(path, exist_ok=True)
    plt.savefig(path + 'supervised_lr_mean_variance.png', bbox_inches='tight', dpi=400)
    df_cross_val.to_csv(path + 'supervised_lr_mean_variance.csv')
    plt.close()


def get_set_transformer_scores(y_val, raw_trains, base_scene):
    scores = []
    # add performance of set transformer
    model_name = 'set_transformer'
    # load set transformer performance
    _out_path = f'output/models/{model_name}/{y_val}_classification/{raw_trains}/{base_scene}/cv/'

    conf = os.listdir(_out_path)
    configs_invariant = []
    configs_ = []
    for c in conf:
        if 'predicted_attributes_permutation_invariant' in c:
            configs_invariant.append(c)
        else:
            configs_.append(c)
    configs = configs_
    configs.insert(0, configs.pop())
    if len(configs) < 3:
        # m_name, num_epochs = 'set_transformer', 150
        # trainer = Trainer(base_scene, raw_trains, device, m_name, num_epochs=num_epochs, X_val=X_val, y_val=y_val, )
        # trainer.cross_val_train()
        raise AssertionError(f'set transformer not trained for all image sizes, available: {configs}')

    for config in configs:
        conf = config.split('_')
        imcount = int(conf[1])
        dir = _out_path + config
        cv_paths = glob.glob(dir + '/*/metrics.json')
        if len(cv_paths) < 5:
            raise AssertionError(f'set transformer cv iteration missing for {config}')
        for iteration, path in enumerate(cv_paths):
            with open(path, 'r') as fp:
                statistics = json.load(fp)
            score = statistics['epoch_acum_accs']['val']['acc'][-1]
            scores.append(['Set Transformer', str(imcount) + ' images', iteration, score])
    return scores


def create_bk(base_scene, num_trains, noise=0.01):
    raw_trains = 'MichalskiTrains'
    y_val = 'direction'
    X_val = 'gt_attributes'
    from m_train_dataset import get_datasets
    ds = get_datasets(base_scene, raw_trains, num_trains, y_val=y_val, X_val=X_val)

    train_c = 0
    path = './output/models/popper/gt/'
    path_dilp = './output/models/dilp/gt/'
    os.makedirs(path, exist_ok=True)
    os.makedirs(path_dilp, exist_ok=True)
    try:
        os.remove(path + '/bk.pl')
    except OSError:
        pass
    try:
        os.remove(path + '/exs.pl')
    except OSError:
        pass
    with open(path + '/exs.pl', 'w+') as exs_file, open(path + '/bk.pl', 'w+') as bk_file, open(
            path_dilp + '/positive.dilp', 'w+') as pos, open(path_dilp + '/negative.dilp', 'w+') as neg:
        for data, label in ds:
            n = np.random.random()
            car_c = 0
            train_c += 1
            data = data.view(-1, 8)
            bk_file.write(f'train(t{train_c}).\n')
            label = 'pos' if label == 1 else 'neg'
            # if train_c < 10:
            exs_file.write(f'{label}(f(t{train_c})).\n')
            if label == 'pos':
                pos.write(f'target(t{train_c}).\n')
            else:
                neg.write(f'target(t{train_c}).\n')
            for car in data:
                # add car to bk if car color is not none
                if car[0] != 0:
                    car_label_names = np.array(ds.attribute_classes)[car.to(dtype=torch.int32).tolist()]
                    color, length, walls, roofs, wheel_count, load_obj1, load_obj2, load_obj3 = car_label_names
                    car_c += 1
                    bk_file.write(f'has_car(t{train_c},t{train_c}_c{car_c}).' + '\n')
                    bk_file.write(f'car_number(t{train_c}_c{car_c},{car_c}).' + '\n')
                    # behind
                    for i in range(1, car_c):
                        bk_file.write(f'behind(t{train_c}_c{car_c},t{train_c}_c{i}).' + '\n')
                    # color
                    bk_file.write(f'{color}(t{train_c}_c{car_c}).' + '\n')
                    # length
                    bk_file.write(f'{length}(t{train_c}_c{car_c}).' + '\n')
                    # walls
                    bk_file.write(f'{walls}(t{train_c}_c{car_c}).' + '\n')
                    # roofs
                    if roofs != 'none':
                        bk_file.write(f'roof_closed(t{train_c}_c{car_c}).' + '\n')
                        bk_file.write(f'{roofs}(t{train_c}_c{car_c}).' + '\n')
                    else:
                        bk_file.write(f'roof_open(t{train_c}_c{car_c}).' + '\n')
                    # wheel_count
                    wheel_num = ['two', 'three'][int(wheel_count[0]) - 2]
                    bk_file.write(f'{wheel_num}{wheel_count[1:]}(t{train_c}_c{car_c}).' + '\n')
                    # payload
                    payload_num = 3 - [load_obj1, load_obj2, load_obj3].count('none')
                    payload_n = ['zero', 'one', 'two', 'three'][payload_num]
                    bk_file.write(f'{payload_n}_load(t{train_c}_c{car_c}).\n')
                    for p_c, payload in enumerate([load_obj1, load_obj2, load_obj3]):
                        if payload != 'none':
                            bk_file.write(f'{payload}(t{train_c}_c{car_c}_l{p_c}).\n')
                            bk_file.write(f'has_load(t{train_c}_c{car_c},t{train_c}_c{car_c}_l{p_c}).\n')

    file = Path(path + '/bk.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    file = Path(path + '/exs.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    shutil.copy(path + '/bk.pl', path_dilp + '/facts.dilp')


def eval_induction_lr(base_scene, raw_trains, image_count):
    y_val = 'direction'
    X_val = 'gt_attributes'
    from m_train_dataset import get_datasets
    full_ds = get_datasets(base_scene, raw_trains, image_count, y_val=y_val, X_val=X_val)
    # train_size, val_size = int(0.7 * image_count), int(0.3 * image_count)
    # train_dataset, val_dataset = torch.utils.data.random_split(full_ds, [train_size, val_size])
    # datasets = {
    #     'train': train_dataset,
    #     'val': val_dataset
    # }
    # y_train = np.concatenate([y for x, y in datasets['train']], axis=0)
    # X_train = np.concatenate([x for x, y in datasets['train']], axis=0)
    # y_val = np.concatenate([y for x, y in datasets['val']], axis=0)
    # X_val = np.concatenate([x for x, y in datasets['val']], axis=0)
    X = np.concatenate([x for x, y in full_ds], axis=0)
    y = np.concatenate([y for x, y in full_ds], axis=0)

    # normalize input data
    scaler = StandardScaler()
    scaler.fit(X)
    X = scaler.transform(X)
    random_state = 1

    # learning relational rules,
    # Examples or target concept may require relational representation that includes multiple entities with relationships between them
    # As you know, first-order logic is a more powerful representation for handling such relational descriptions
    # Consistency, Completeness
    # pruning?
    print(f'supervised learning methods on michalski train attributes to predict the corresponding labels')
    print(f'the methods described below were employed on a michalski train dataset of {image_count} images')
    print(f'the score represents the methods mean accuracy over 5 folds of cross-validation')
    print('-' * 10)

    # Gaussian Naive Bayes
    clf = GaussianNB()
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'Gaussian Naive Bayes score: {round(acc * 100, 2)}%')
    print('-' * 10)

    # stochastic gradient descent
    clf = SGDClassifier()
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'stochastic gradient descent score: {round(acc * 100, 2)}%')
    print('-' * 10)

    # support vector machine
    clf = svm.SVC()
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'Support Vector Machine score: {round(acc * 100, 2)}%')
    print('-' * 10)

    # K-Neighbors Classifier
    knn = KNeighborsClassifier(n_neighbors=10, weights="distance")
    scores = cross_val_score(knn, X, y, cv=5)
    acc = scores.mean()
    print(f'K-Neighbors Classifier score: {round(acc * 100, 2)}%')
    print('-' * 10)

    # DecisionTreeClassifier
    clf = DecisionTreeClassifier()
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'Decision Tree Classifier score: {round(acc * 100, 2)}%')

    # forrest = RandomForestClassifier(random_state=random_state)
    # grid_params = {
    #     'n_estimators': [50, 100, 200],
    #     'max_depth': [1, 3, 5, 8, 10, None],
    #     'min_samples_leaf': [3, 6, 10, 13, 16, 20]
    # }
    # clf = GridSearchCV(forrest, grid_params, cv=5)
    # clf.fit(X, y)
    # best_params = clf.best_params_
    # forrest = RandomForestClassifier(n_estimators=best_params['n_estimators'], max_depth=best_params['max_depth'],
    #                                  min_samples_leaf=best_params['min_samples_leaf'], random_state=random_state)
    forrest = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=2,
                                     random_state=random_state)
    scores = cross_val_score(forrest, X, y, cv=5)
    acc = scores.mean()
    print(f'RandomForestClassifier score: {round(acc * 100, 2)}%')

    clf = ExtraTreesClassifier(n_estimators=10, max_depth=None, min_samples_split=2, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'ExtraTreesClassifier score: {round(acc * 100, 2)}%')
    print('-' * 10)

    clf = AdaBoostClassifier(n_estimators=100)
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'AdaBoostClassifier score: {round(acc * 100, 2)}%')

    clf = GradientBoostingClassifier(n_estimators=500, learning_rate=1, max_depth=3, random_state=random_state)
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'GradientBoostingClassifier score: {round(acc * 100, 2)}%')

    clf = HistGradientBoostingClassifier(max_iter=100)
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'HistGradientBoostingClassifier score: {round(acc * 100, 2)}%')
    print('-' * 10)

    clf = MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(15, 200), random_state=random_state,
                        max_iter=1000)
    scores = cross_val_score(clf, X, y, cv=5)
    acc = scores.mean()
    print(f'multi-layer perceptron score: {round(acc * 100, 2)}%')
    print('-' * 10)

    # used github repo https://github.com/imoscovitz/wittgenstein
    # https://towardsdatascience.com/how-to-perform-explainable-machine-learning-classification-without-any-trees-873db4192c68
    # Foil Gain Metric / First order inductive logic
    import wittgenstein as lw
    ripper_clf = lw.RIPPER(random_state=random_state)
    scores = cross_val_score(ripper_clf, X, y, cv=5)
    acc = scores.mean()
    # ripper_clf.fit(X_train, y_train)  # Or pass X and y data to .fit
    # acc = ripper_clf.score(X_val, y_val)
    print(f'ripper_clf score: {round(acc * 100, 2)}%')

    irep_clf = lw.IREP(random_state=random_state)
    scores = cross_val_score(irep_clf, X, y, cv=5)
    acc = scores.mean()
    print(f'irep_clf score: {round(acc * 100, 2)}%')

    # stacking classifier
    # tree = DecisionTreeClassifier(random_state=random_state)
    # nb = GaussianNB()
    # estimators = [("rip", ripper_clf), ("tree", tree), ("nb", nb)]
    # ensemble_clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    # scores = cross_val_score(ensemble_clf, X, y, cv=5)
    # acc = scores.mean()
    # print(f'Ensemble classifier accuracy: {round(acc * 100, 2)}%')


def analyse_ds(ds):
    train_c = 0
    eastbound_trains = 0
    trains_with_closed_and_short_cars_heading_east = 0
    trains_with_closed_and_short_cars_heading_west = 0
    trains_with_circular_and_triangular_east = 0
    trains_with_circular_and_triangular_west = 0
    trains_with_a_closed_short_car = 0
    trains_with_circular_behind_triangular = 0
    trains_with_both = 0
    noise_east = 0
    noise_west = 0

    for item_ind in range(ds.__len__()):
        train = ds.get_m_train(item_ind)
        train_c += 1
        dir_label = train.get_label()
        short_car = False
        closed_car = False
        circular_load = False
        circular_load_pos = None
        triangular_load = False
        triangular_load_pos = None

        circular_behind_triangular = False
        short_closed_car = False
        eastbound_trains += 1 if dir_label == 'east' else 0

        for car in train.get_cars():
            c_num = car.get_car_number()
            if car.get_car_length() == 'short':
                short_car = True
                if car.get_car_roof() in ["arc", "flat", "jagged", "peaked"]:
                    short_closed_car = True

            if car.get_car_roof() in ["arc", "flat", "jagged", "peaked"]:
                closed_car = True

            if car.get_load_number() > 0:
                if car.get_load_shape() == "circle":
                    circular_load = True
                    circular_load_pos = c_num
                if car.get_load_shape() == "triangle":
                    triangular_load = True
                    if triangular_load_pos is None:
                        triangular_load_pos = c_num

        if closed_car and short_car:
            if dir_label == 'east':
                trains_with_closed_and_short_cars_heading_east += 1
            else:
                trains_with_closed_and_short_cars_heading_west += 1
        if short_closed_car:
            trains_with_a_closed_short_car += 1
        if circular_load and triangular_load:
            if dir_label == 'east':
                trains_with_circular_and_triangular_east += 1
            else:
                trains_with_circular_and_triangular_west += 1
            if triangular_load_pos < circular_load_pos:
                circular_behind_triangular = True
                trains_with_circular_behind_triangular += 1
                if short_closed_car:
                    trains_with_both += 1
        if dir_label == 'east':
            if not circular_behind_triangular and not short_closed_car:
                noise_east += 1
                # raise AssertionError('train with label east does not fulfill rule')
        if circular_behind_triangular or short_closed_car:
            if dir_label == 'west':
                noise_west += 1
                # raise AssertionError('train fulfills rule but is labeled as west')

    westbound_trains = train_c - eastbound_trains
    print(f'number of trains is {train_c}')
    print(f'number of trains heading east {eastbound_trains} ({round(100 * eastbound_trains / train_c)}%)')
    print(f'number of trains heading west {westbound_trains} ({round(100 * westbound_trains / train_c)}%)')
    print('Eastbound trains must comply the following rule: '
          'There is either a short, closed car, '
          'or a car with a circular load somewhere behind a car with a triangular load.\n')

    print(f'number of trains with a short car and a closed car: '
          f'{trains_with_closed_and_short_cars_heading_east + trains_with_closed_and_short_cars_heading_west} '
          f'({round(((trains_with_closed_and_short_cars_heading_east + trains_with_closed_and_short_cars_heading_west) / train_c) * 100)}%)\n'

          f'eastbound trains with a short car and a closed car: {trains_with_closed_and_short_cars_heading_east} '
          f'({round(100 * trains_with_closed_and_short_cars_heading_east / (trains_with_closed_and_short_cars_heading_east + trains_with_closed_and_short_cars_heading_west))}%), '
          f'westbound trains {trains_with_closed_and_short_cars_heading_west} '
          f'({round(100 * trains_with_closed_and_short_cars_heading_west / (trains_with_closed_and_short_cars_heading_east + trains_with_closed_and_short_cars_heading_west))}%)\n')

    print(f'number of trains with a short closed car: '
          f'{trains_with_a_closed_short_car} ({round((trains_with_a_closed_short_car / train_c) * 100)}%) '
          f'representing {round((trains_with_a_closed_short_car / eastbound_trains) * 100)}% of all eastbound trains\n'

          f'Since short closed car within a michalski train is a strict indicator for an eastbound train \n'
          # f'If the underlying decision rule would be known a classification algorithm'
          f'Only using this classification indicator an algorithm could achieve a maximum accuracy of: '
          f'{round(((trains_with_a_closed_short_car + westbound_trains) / train_c) * 100)}% \n')

    print(
        f'number of trains having loaded circular and triangular loads: '
        f'{trains_with_circular_and_triangular_east + trains_with_circular_and_triangular_west} '
        f'({round(((trains_with_circular_and_triangular_east + trains_with_circular_and_triangular_west) / train_c) * 100)}%)\n'

        f'eastbound trains with a circular and triangular loads: {trains_with_circular_and_triangular_east} '
        f'({round(100 * trains_with_circular_and_triangular_east / (trains_with_circular_and_triangular_east + trains_with_circular_and_triangular_west))}%) '
        f'westbound trains {trains_with_circular_and_triangular_west} '
        f'({round(100 * trains_with_circular_and_triangular_west / (trains_with_circular_and_triangular_east + trains_with_circular_and_triangular_west))})%\n')

    print(f'number of trains with a circular load behind a triangular load: '
          f'{trains_with_circular_behind_triangular} ({round((trains_with_circular_behind_triangular / train_c) * 100)}%) '
          f'representing {round((trains_with_circular_behind_triangular / eastbound_trains) * 100)}% of all eastbound trains\n'

          f'Since a circular load behind a triangular load within a michalski train is a strict indicator for an eastbound train \n'
          # f'If the underlying decision rule would be known a classification algorithm'
          f'Only using this classification indicator an algorithm could achieve a maximum accuracy of: '
          f'{round(((trains_with_circular_behind_triangular + westbound_trains) / train_c) * 100)}% \n')

    print(f'number of trains with a circular load behind a triangular load and a short closed car: {trains_with_both}')

    print(f'number of trains marked as noise east: {noise_east}')
    print(f'number of trains marked as noise west: {noise_west}')

def ds_attr_pred_confusion_matrix():
    scenes = ['base_scene', 'desert_scene', 'sky_scene', 'fisheye_scene']

    X_val = 'predicted_attributes'
    y_val = 'attribute'
    raw_trains = 'MichalskiTrains'
    path = 'output/predicted_ds'
    os.makedirs(path, exist_ok=True)
    for scene in scenes:
        import m_train_dataset
        ds = m_train_dataset.get_datasets(scene, raw_trains, 10000, y_val, X_val=X_val)
        gt_attr = np.concatenate([y for x, y in ds], axis=0).flatten()
        for image_count in [100, 1000, 8000]:
            ds.predictions_im_count = image_count
            predicted_attr = np.concatenate([x for x, y in ds], axis=0).flatten()
            ConfusionMatrixDisplay.from_predictions(gt_attr, predicted_attr, include_values=False,
                                                    cmap=plt.cm.Blues,
                                                    normalize='true')
            # plt.title(f'Confusion Matrix for {model_name} (acc: {round(acc * 100, 2)}%)\n'
            #           f'trained on {raw_trains} in {base_scene} DS')
            plt.savefig(path + f'/{scene}_{image_count}_ims_confusion_matrix', dpi=400, bbox_inches='tight')
            plt.close()