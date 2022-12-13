import os
import shutil
from itertools import product
from pathlib import Path
import random
import numpy as np
import torch
from sklearn.model_selection import StratifiedShuffleSplit

from raw.gen_raw_trains import read_trains


def create_datasets(rules, num_samples, train_description, folds, ds_total_size, noise_vals, replace_old=True):
    print(f'preparing {folds} fold cross-val {train_description} datasets for {rules} rules '
          f'with sample sizes of {num_samples} and noises of {noise_vals}')
    gen_ds = 0
    for num_tsamples, rule, noise in product(num_samples, rules, noise_vals):
        ds_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{train_description}_{rule}.txt'
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
                if not (not replace_old
                        and os.path.exists(f'{out_path}/aleph/trains2/train.b')
                        and os.path.exists(f'{out_path}/aleph/trains2/train.f')
                        and os.path.exists(f'{out_path}/aleph/trains2/train.n')
                        and os.path.exists(f'{out_path}/popper/gt1/bias.pl')
                        and os.path.exists(f'{out_path}/popper/gt1/bk.pl')
                        and os.path.exists(f'{out_path}/popper/gt1/exs.pl')):
                    create_bk(train_path, out_path, num_tsamples, noise)
                    gen_ds += 1
    n_ds = len(num_samples)*len(rules)*len(noise_vals)*folds
    print(f'total of {n_ds} ds: found {n_ds-gen_ds} existing ds, generated {gen_ds} remaining ds')


def create_bk(ds_path, out_path, ds_size=None, noise=0):
    train_c = 0
    path_1 = f'{out_path}/popper/gt1'
    path_2 = f'{out_path}/popper/gt2'
    path_3 = f'{out_path}/popper/gt3'
    path_dilp = f'{out_path}/dilp/gt'
    path_aleph = f'{out_path}/aleph/trains2'
    paths = [path_1, path_2, path_3, path_aleph, path_dilp]
    for path in paths:
        os.makedirs(path, exist_ok=True)
        for file in [path + '/bk.pl', path + '/exs.pl']:
            try:
                os.remove(file)
            except OSError:
                pass
    for file in [path_aleph + '/train.n', path_aleph + '/train.f']:
        try:
            os.remove(file)
        except OSError:
            pass

    trains = read_trains(ds_path)
    with open(path_1 + '/exs.pl', 'w+') as exs_file, open(path_1 + '/bk.pl', 'w+') as bk_file, \
            open(path_2 + '/bk.pl', 'w+') as bk2_file, open(path_3 + '/bk.pl', 'w+') as bk3_file, open(
        path_dilp + '/positive.dilp', 'w+') as pos, open(path_dilp + '/negative.dilp', 'w+') as neg, open(
        path_aleph + '/train.f', 'w+') as aleph_pos, open(path_aleph + '/train.n', 'w+') as aleph_neg:
        ds_size = len(trains) if ds_size is None else ds_size
        if len(trains) < ds_size:
            raise AssertionError(f'not enough trains in DS {len(trains)} to create a bk of size {ds_size}')
        for train in trains[:ds_size]:
            ns = random.random()
            label = train.get_label()
            train_c += 1
            bk_file.write(f'train(t{train_c}).\n')
            bk2_file.write(f'train(t{train_c}).\n')
            bk3_file.write(f'train(t{train_c}).\n')
            label = 'pos' if label == 'east' else 'neg'
            # if train_c < 10:
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
                if ns < noise:
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
                    color = car.get_blender_car_color()
                    length = car.get_car_length()
                    walls = car.get_blender_wall()
                    roofs = car.get_blender_roof()
                    wheel_count = car.get_car_wheels()
                    l_shape = car.get_blender_payload()
                    l_num = car.get_load_number()
                    load_obj1, load_obj2, load_obj3 = [l_shape] * l_num + ['none'] * (3 - l_num)
                    car_number = car.get_car_number()

                bk_file.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                bk_file.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                bk2_file.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                bk3_file.write(f'has_car(t{train_c},t{train_c}_c{car_number}).' + '\n')
                bk3_file.write(f'car_num(t{train_c}_c{car_number},{car_number}).' + '\n')
                position = ['first', 'second', 'third', 'fourth']
                # bk_file.write(f'{position[car_number - 1]}_car(t{train_c}_c{car_number}).' + '\n')
                # # behind
                # for i in range(1, car_c):
                #     bk_file.write(f'behind(t{train_c}_c{car_c},t{train_c}_c{i}).' + '\n')
                # color
                bk_file.write(f'{color}(t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'{color}3(t{train_c}_c{car_number}_color).' + '\n')
                bk2_file.write(f'car_color3(t{train_c}_c{car_number},t{train_c}_c{car_number}_color).' + '\n')
                bk3_file.write(f'car_color(t{train_c}_c{car_number},{color}).' + '\n')
                # length
                bk_file.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                bk3_file.write(f'{length}(t{train_c}_c{car_number}).' + '\n')
                # walls
                bk_file.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                bk2_file.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                bk3_file.write(f'{walls}(t{train_c}_c{car_number}).' + '\n')
                # roofs
                if roofs != 'none':
                    #     bk_file.write(f'roof_closed(t{train_c}_c{car_number}).' + '\n')
                    bk_file.write(f'{roofs}(t{train_c}_c{car_number}).' + '\n')
                    bk2_file.write(f'{roofs}3(t{train_c}_c{car_number}_roof).' + '\n')
                else:
                    bk_file.write(f'roof_open(t{train_c}_c{car_number}).' + '\n')
                    bk2_file.write(f'roof_open3(t{train_c}_c{car_number}_roof).' + '\n')

                bk2_file.write(f'has_roof3(t{train_c}_c{car_number},t{train_c}_c{car_number}_roof).' + '\n')
                bk3_file.write(f'has_roof2(t{train_c}_c{car_number},{roofs}).' + '\n')

                # wheel_count
                wheel_num = ['two', 'three'][int(wheel_count[0]) - 2]
                bk_file.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')
                bk2_file.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')
                bk3_file.write(f'has_wheel0(t{train_c}_c{car_number},{wheel_count[0]}).' + '\n')

                # payload
                payload_num = 3 - [load_obj1, load_obj2, load_obj3].count('none')
                payload_n = ['zero', 'one', 'two', 'three'][payload_num]
                bk_file.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')
                bk2_file.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')
                bk3_file.write(f'load_num(t{train_c}_c{car_number},{l_num}).\n')

                if l_num > 0:
                    bk2_file.write(f'{l_shape}3(t{train_c}_c{car_number}_payload).\n')
                    bk2_file.write(f'has_payload3(t{train_c}_c{car_number},t{train_c}_c{car_number}_payload).\n')
                bk3_file.write(f'has_payload(t{train_c}_c{car_number},{l_shape}).\n')
                for p_c, payload in enumerate([load_obj1, load_obj2, load_obj3]):
                    if payload != 'none':
                        bk_file.write(f'{payload}(t{train_c}_c{car_number}).\n')
                        # bk_file.write(f'has_payload3(t{train_c}_c{car_number},t{train_c}_c{car_number}_l{p_c}).\n')

    file = Path(path_1 + '/bk.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    file = Path(path_2 + '/bk.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    file = Path(path_3 + '/bk.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    file = Path(path_1 + '/exs.pl')
    file.write_text(
        "\n".join(
            sorted(
                file.read_text().split("\n")[:-1]
            )
        )
    )
    try:
        os.remove(path_aleph + '/train.b')
    except OSError:
        pass
    with open('ilp/aleph/trains2/bias3', 'r') as bias, open(path_3 + '/bk.pl', 'r') as bk, open(path_aleph + '/train.b',
                                                                                                'w+') as comb:
        comb.write(bias.read() + '\n')
        comb.write(bk.read())
    shutil.copy('ilp/popper/gt1/bias.pl', path_1 + '/bias.pl')
    shutil.copy('ilp/popper/gt2/bias.pl', path_2 + '/bias.pl')
    shutil.copy('ilp/popper/gt3/bias.pl', path_3 + '/bias.pl')

    shutil.copy(path_1 + '/bk.pl', path_dilp + '/facts.dilp')
    shutil.copy(path_1 + '/exs.pl', path_2 + '/exs.pl')
    shutil.copy(path_1 + '/exs.pl', path_3 + '/exs.pl')
