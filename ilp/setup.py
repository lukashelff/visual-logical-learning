import os
import shutil
from itertools import product
from pathlib import Path
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from raw.gen_raw_trains import read_trains


def create_datasets(rules, num_samples, train_description, folds, ds_total_size, noise_vals, replace_existing=True,
                    min_cars=2, max_cars=4, tag='',
                    tg_output_path=f'TrainGenerator/output'
                    ):
    print(f'preparing {folds} fold cross-val {train_description} datasets for {rules} rules '
          f'with sample sizes of {num_samples} and noises of {noise_vals}')
    gen_ds = 0
    for num_tsamples, rule, noise in product(num_samples, rules, noise_vals):
        #      ds_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{train_description}_{rule}.txt'
        ds_path = f'{tg_output_path}/image_generator/dataset_descriptions/{rule}/{tag}{train_description}_len_{min_cars}-{max_cars}.txt'

        with open(ds_path, "r") as file:
            all_data = file.readlines()
            if len(all_data) != ds_total_size:
                raise f'datasets of size {ds_total_size} however only {len(all_data)} datasamples were generated'
            y = [l[0] for l in all_data]
            sss = StratifiedShuffleSplit(n_splits=folds, train_size=num_tsamples, test_size=2000,
                                         random_state=0)
            for fold, (tr_idx, val_idx) in enumerate(sss.split(np.zeros(len(y)), y)):
                out_path = f'output/ilp/datasets/{rule}/{train_description}{num_tsamples}_{noise}noise/cv_{fold}'
                os.makedirs(out_path, exist_ok=True)
                train_path = f'{out_path}/train_samples.txt'
                val_path = f'{out_path}/val_samples.txt'
                train_samples = map(all_data.__getitem__, tr_idx)
                val_samples = map(all_data.__getitem__, val_idx)
                for ds in [train_path, val_path]:
                    try:
                        os.remove(ds)
                    except OSError:
                        pass
                with open(train_path, 'w+') as train, open(val_path, 'w+') as val:
                    train.writelines(train_samples)
                    val.writelines(val_samples)
                if not (not replace_existing
                        and os.path.exists(f'{out_path}/aleph/trains2/train.b')
                        and os.path.exists(f'{out_path}/aleph/trains2/train.f')
                        and os.path.exists(f'{out_path}/aleph/trains2/train.n')
                        and os.path.exists(f'{out_path}/popper/gt1/bias.pl')
                        and os.path.exists(f'{out_path}/popper/gt1/bk.pl')
                        and os.path.exists(f'{out_path}/popper/gt1/exs.pl')):
                    create_bk(train_path, out_path, num_tsamples, noise)
                    gen_ds += 1
    n_ds = len(num_samples) * len(rules) * len(noise_vals) * folds
    print(f'total of {n_ds} ds: found {n_ds - gen_ds} existing ds, generated {gen_ds} remaining ds')


def create_bk(ds_path, out_path, ds_size=None, noise=0, noise_type='label'):
    ''' creates a bk for a given dataset, ds_size is the number of trains to use for the bk
    noise is the percentage of noise to add to the bk
    noise_type is the type of noise to add, can be 'label' or 'attribute'

    the bk is created in the following format:
    popper: bk.pl, exs.pl, bias.pl
    aleph: train.b, train.f, train.n
    dilp: positive.dilp, negative.dilp, facts.dilp

    the bk is created by randomly selecting ds_size trains from the dataset and adding them to the bk
    if noise is not 0, noise% of the trains are randomly selected and their labels are flipped
    if noise_type is 'attribute', noise% of the trains are randomly selected and their attributes are flipped


    '''
    train_c = 0
    path_1 = f'{out_path}/popper/gt1'
    path_2 = f'{out_path}/popper/gt2'
    path_3 = f'{out_path}/popper/gt3'
    path_dilp = f'{out_path}/dilp/gt'
    path_aleph = f'{out_path}/aleph/trains2'
    paths = [path_1, path_2, path_3, path_aleph, path_dilp]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        if 'aleph' in path:
            dirs = ['/train.n', '/train.f', '/train.b']
        elif 'popper' in path:
            dirs = ['/bk.pl', '/exs.pl', '/bias.pl']
        elif 'dilp' in path:
            dirs = ['/facts.dilp', '/negative.dilp', '/positive.dilp']
        for file in dirs:
            try:
                open(path + file, 'w').close()
            except OSError:
                pass

    trains = read_trains(ds_path)
    with open(path_1 + '/exs.pl', 'w+') as exs_file, open(path_1 + '/bk.pl', 'w+') as popper_bk, \
            open(path_2 + '/bk.pl', 'w+') as popper_bk2, open(path_3 + '/bk.pl', 'w+') as popper_bk3, open(
        path_dilp + '/positive.dilp', 'w+') as pos, open(path_dilp + '/negative.dilp', 'w+') as neg, open(
        path_aleph + '/train.f', 'w+') as aleph_pos, open(path_aleph + '/train.n', 'w+') as aleph_neg, open(
        path_aleph + '/train.b', 'w+') as aleph_bk, open('ilp/aleph/trains2/bias2', 'r') as aleph_bias:
        ds_size = len(trains) if ds_size is None else ds_size
        aleph_bk.write(aleph_bias.read() + '\n')
        if len(trains) < ds_size:
            raise AssertionError(f'not enough trains in DS {len(trains)} to create a bk of size {ds_size}')
        for train in trains[:ds_size]:
            ns = random.random()
            label = train.get_label()
            train_c += 1
            popper_bk.write(f'train(t{train_c}).\n')
            popper_bk2.write(f'train(t{train_c}).\n')
            popper_bk3.write(f'train(t{train_c}).\n')
            label = 'pos' if label == 'east' else 'neg'
            if ns < noise and noise_type == 'label':
                label = 'pos' if label == 'neg' else 'neg'
            exs_file.write(f'{label}(eastbound(t{train_c})).\n')

            if label == 'pos':
                pos.write(f'target(t{train_c}).\n')
                aleph_pos.write(f'eastbound(t{train_c}).\n')
            else:
                neg.write(f'target(t{train_c}).\n')
                aleph_neg.write(f'eastbound(t{train_c}).\n')
            for car in train.get_cars():

                # add car to bk if car color is not none
                # car_label_names = np.array(ds_val.attribute_classes)[car.to(dtype=torch.int32).tolist()]
                # color, length, walls, roofs, wheel_count, load_obj1, load_obj2, load_obj3 = car_label_names
                if ns < noise and noise_type == 'attribute':
                    car_number = car.get_car_number()
                    color = ['yellow', 'green', 'grey', 'red', 'blue'][np.random.randint(5)]
                    length = ['short', 'long'][np.random.randint(2)]
                    walls = ["braced_wall", 'solid_wall'][np.random.randint(2)]
                    roofs = ["roof_foundation", 'solid_roof', 'braced_roof', 'peaked_roof'][np.random.randint(4)]
                    wheel_count = ['2_wheels', '3_wheels'][np.random.randint(2)]
                    l_shape = ["box", "golden_vase", 'barrel', 'diamond', 'metal_pot', 'oval_vase'][
                        np.random.randint(6)]
                    l_num = np.random.randint(4)
                    load_obj1, load_obj2, load_obj3 = [l_shape] * l_num + ['none'] * (3 - l_num)
                else:
                    car_number = car.get_car_number()
                    color = car.get_blender_car_color()
                    length = car.get_car_length()
                    walls = car.get_blender_wall()
                    roofs = car.get_blender_roof()
                    wheel_count = car.get_car_wheels()
                    l_shape = car.get_blender_payload()
                    l_num = car.get_load_number()
                    load_obj1, load_obj2, load_obj3 = [l_shape] * l_num + ['none'] * (3 - l_num)

                payload_num = 3 - [load_obj1, load_obj2, load_obj3].count('none')
                payload_n = ['zero', 'one', 'two', 'three'][payload_num]
                wheel_num = ['two', 'three'][int(wheel_count[0]) - 2]

                popper_bk.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                popper_bk.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                popper_bk2.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                popper_bk2.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                popper_bk3.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                popper_bk3.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                aleph_bk.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                aleph_bk.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')

                # color
                popper_bk.write(f'{color}(t{train_c}_c{car_number}).' + '\n')
                popper_bk2.write(f'{color}3(t{train_c}_c{car_number}_color).' + '\n')
                popper_bk2.write(f'car_color3(t{train_c}_c{car_number},t{train_c}_c{car_number}_color).' + '\n')
                popper_bk3.write(f'{color}(t{train_c}_c{car_number}).' + '\n')
                aleph_bk.write(f'car_color(t{train_c}_c{car_number},{color}).' + '\n')

                # length
                popper_bk.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                popper_bk2.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                popper_bk3.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                aleph_bk.write(f'{length}(t{train_c}_c{car_number}).' + '\n')

                # walls
                popper_bk.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                popper_bk2.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                popper_bk3.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                aleph_bk.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')

                # roofs
                if roofs != 'none':
                    #     bk_file.write(f'roof_closed(t{train_c}_c{car_number}).' + '\n')
                    popper_bk.write(f'{roofs}(t{train_c}_c{car_number}).' + '\n')
                    popper_bk2.write(f'{roofs}3(t{train_c}_c{car_number}_roof).' + '\n')
                    popper_bk3.write(f'{roofs}(t{train_c}_c{car_number}).' + '\n')
                else:
                    popper_bk.write(f'roof_open(t{train_c}_c{car_number}).' + '\n')
                    popper_bk2.write(f'roof_open3(t{train_c}_c{car_number}_roof).' + '\n')
                    popper_bk3.write(f'roof_open(t{train_c}_c{car_number}).' + '\n')
                popper_bk2.write(f'has_roof3(t{train_c}_c{car_number},t{train_c}_c{car_number}_roof).' + '\n')
                aleph_bk.write(f'has_roof2(t{train_c}_c{car_number},{roofs}).' + '\n')

                # wheel_count
                popper_bk.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')
                popper_bk2.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')
                popper_bk3.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')
                aleph_bk.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')

                # payload number
                popper_bk.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')
                popper_bk2.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')
                popper_bk3.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')
                aleph_bk.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')

                # payload
                if l_num > 0:
                    popper_bk2.write(f'{l_shape}3(t{train_c}_c{car_number}_payload).\n')
                    popper_bk2.write(f'has_payload3(t{train_c}_c{car_number},t{train_c}_c{car_number}_payload).\n')
                    popper_bk.write(f'{l_shape}(t{train_c}_c{car_number}).\n')
                    aleph_bk.write(f'has_payload(t{train_c}_c{car_number},{l_shape}).\n')

                for p_c in range(l_num):
                    popper_bk3.write(f'{l_shape}3(t{train_c}_c{car_number}_l{p_c}).\n')
                    popper_bk3.write(f'has_payload3(t{train_c}_c{car_number},t{train_c}_c{car_number}_l{p_c}).\n')
                # for p_c, payload in enumerate([load_obj1, load_obj2, load_obj3]):
                #     if payload != 'none':
                #         bk3_file.write(f'{l_shape}3(t{train_c}_c{car_number}_l{p_c}).\n')
                #         bk3_file.write(f'has_payload3(t{train_c}_c{car_number},t{train_c}_c{car_number}_l{p_c}).\n')
                #         bk_file.write(f'{payload}(t{train_c}_c{car_number}).\n')

    sort_file(Path(path_1 + '/bk.pl'))
    sort_file(Path(path_2 + '/bk.pl'))
    sort_file(Path(path_3 + '/bk.pl'))
    sort_file(Path(path_1 + '/exs.pl'))
    # with open('ilp/aleph/trains2/bias2', 'r') as aleph_bias, open(path_aleph + '/train.b', 'r') as aleph_bk, open(
    #         path_aleph + '/tmp', 'w+') as tmp:
    #     tmp.write(aleph_bias.read() + '\n')
    #     tmp.write(aleph_bk.read())
    # os.remove(path_aleph + '/train.b')
    # os.rename('tmp', path_aleph + '/train.b')

    shutil.copy('ilp/popper/gt1/bias.pl', path_1 + '/bias.pl')
    shutil.copy('ilp/popper/gt2/bias.pl', path_2 + '/bias.pl')
    shutil.copy('ilp/popper/gt3/bias.pl', path_3 + '/bias.pl')

    shutil.copy(path_1 + '/bk.pl', path_dilp + '/facts.dilp')
    shutil.copy(path_1 + '/exs.pl', path_2 + '/exs.pl')
    shutil.copy(path_1 + '/exs.pl', path_3 + '/exs.pl')


def sort_file(file):
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )


def setup_alpha_ilp_ds(base_scene, raw_trains, train_vis, ds_size, ds_path, class_rule):
    import shutil
    from michalski_trains.dataset import get_datasets
    ds = get_datasets(base_scene, raw_trains, train_vis, ds_size, ds_path=ds_path, class_rule=class_rule)
    path_train_true = f'output/alphailp-images/{class_rule}/train/true'
    path_test_true = f'output/alphailp-images/{class_rule}/test/true'
    path_val_true = f'output/alphailp-images/{class_rule}/val/true'
    path_train_false = f'output/alphailp-images/{class_rule}/train/false'
    path_test_false = f'output/alphailp-images/{class_rule}/test/false'
    path_val_false = f'output/alphailp-images/{class_rule}/val/false'
    for p in [path_train_true, path_test_true, path_val_true, path_train_false, path_test_false, path_val_false]:
        os.makedirs(p, exist_ok=True)
        p += '/image'
    train_t, test_t, val_t, train_f, test_f, val_f = [0] * 6
    c = 0
    while sum([train_t, test_t, val_t, train_f, test_f, val_f]) < 300:
        path = ds.get_image_path(c)
        label = ds.get_label_for_id(c)
        # print(f'iteration: {c}, label: {label}, path: {path}')
        if label == 'east':
            if train_t < 50:
                shutil.copyfile(path, path_train_true + f'/image{train_t}.png')
                train_t += 1
            elif test_t < 50:
                shutil.copyfile(path, path_test_true + f'/image{test_t}.png')
                test_t += 1
            elif val_t < 50:
                shutil.copyfile(path, path_val_true + f'/image{val_t}.png')
                val_t += 1
        elif label == 'west':
            if train_f < 50:
                shutil.copyfile(path, path_train_false + f'/image{train_f}.png')
                train_f += 1
            elif test_f < 50:
                shutil.copyfile(path, path_test_false + f'/image{test_f}.png')
                test_f += 1
            elif val_f < 50:
                shutil.copyfile(path, path_val_false + f'/image{val_f}.png')
                val_f += 1
        c += 1
