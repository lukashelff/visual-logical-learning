import multiprocessing
import multiprocessing as mp
import os
import random
import time
from itertools import product
from multiprocessing import Process
from typing import Type
import warnings
import pandas as pd
from pebble import ProcessPool

from ilp import visualization
from ilp.dataset_functions import create_bk, create_cv_datasets
from raw.concept_tester import eval_rule
from rdm.wrappers import Aleph


class Ilp_trainer():

    def __int__(self):
        self.model, self.num_tsamples, self.rule, self.noise = [None] * 4

    # cmd = f"echo \"read_all(trains2/train). induce.\" | yap -s5000 -h20000 -l aleph.pl > log.txt 2>&1"
    def cross_val(self, train_description, rules=['numerical', 'theoryx', 'complex'], models=['aleph', 'popper'],
                  folds=5, train_count=[100, 1000, 10000], ds_size=12000, noise=0, complete_run=True, log=False,
                  output_dir='output/ilp', tag='', min_cars=2, max_cars=4, per_run_timeout=60 * 60 * 24,
                  symbolic_ds_path=f'TrainGenerator/output/image_generator/dataset_descriptions'):
        ilp_stats_path = output_dir + '/stats'
        ds_path = output_dir + f'/datasets'
        noise_vals = noise if type(noise) is list else [noise]
        create_cv_datasets(rules, train_count, train_description, folds, ds_size, noise_vals, replace_existing=False,
                           output_dir=ds_path, symbolic_ds_path=symbolic_ds_path, tag=tag, min_cars=min_cars,
                           max_cars=max_cars)
        os.makedirs(ilp_stats_path, exist_ok=True)
        t_its = len(models) * len(train_count) * len(rules) * len(noise_vals)

        for it, (model, num_tsamples, rule, noise) in enumerate(product(models, train_count, rules, noise_vals)):
            self.model, self.num_tsamples, self.rule, self.noise = model, num_tsamples, rule, noise
            txt = f'iteration ({it}/{t_its}): {model} learning {rule} rule: {folds}-fold Cross-Validation' \
                  f' with {num_tsamples} training samples with {noise * 100}% noise with a timeout of {per_run_timeout}'
            txt += tag if tag == '' else f' with tag {tag}'
            print(txt)

            csv = f'{ilp_stats_path}/{tag}{model}_{rule}_{num_tsamples}smpl_{noise}noise.csv' if noise > 0 else \
                f'{ilp_stats_path}/{tag}{model}_{rule}_{num_tsamples}smpl.csv'
            # if complete_run and os.path.exists(csv) and not pd.read_csv(open(csv)).empty:
            if complete_run and os.path.exists(csv):
                print('found training results of previous run, skipping training')
            else:
                try:
                    os.remove(csv)
                except:
                    pass
                inputs = [f'{ds_path}/{rule}/{tag}{train_description}{num_tsamples}_{noise}noise/cv_{fold}'
                          for fold in range(folds)]
                # self.popper_train(out_path)
                if model == 'popper':
                    worker = 5

                    # with mp.Pool(5) as p:
                    #     inputs = [(i, log) for i_c, i in enumerate(inputs)]
                    #     out = p.starmap(self.popper_train, inputs)
                    futures = []
                    with ProcessPool(worker) as p:
                        out = [(None, None)] * worker
                        for i in inputs:
                            # prev timout was at 60 * 60 * 3, 3 hours
                            futures.append(p.schedule(self.popper_train, args=[i, log], timeout=per_run_timeout))
                        for c, f in enumerate(futures):
                            try:
                                out[c] = f.result()
                            except:
                                pass
                elif model == 'aleph':
                    # out = []
                    # for c, input in enumerate(inputs):
                    #     out.append(self.aleph_train(input, print_stats=True))
                    worker = 2 if num_tsamples > 1000 else 5
                    with mp.Pool(worker) as p:
                        inputs = [(i, noise * num_tsamples, log, per_run_timeout) for i_c, i in enumerate(inputs)]
                        out = p.starmap(self.aleph_train, inputs)
                else:
                    raise ValueError(f'model: {model} not supported')
                results = self.to_df(out)
                if results.empty:
                    failed_csv = csv.replace('.csv', '_failed.csv')
                    warnings.warn(f'{model} exited with no results. Empty stats saved to {failed_csv}.')
                    results.to_csv(failed_csv)
                else:
                    results.to_csv(csv)
                    print(f'saved statistics in {csv}')

    def train(self, model, raw_trains, class_rule, train_size, val_size, noise=0, train_log=False):
        ds_path = f'TrainGenerator/output/image_generator/dataset_descriptions/{raw_trains}_{class_rule}.txt'
        out_path = f'output/ilp/datasets/train'
        os.makedirs(out_path, exist_ok=True)
        train_path = f'{out_path}/train_samples.txt'
        val_path = f'{out_path}/val_samples.txt'
        with open(ds_path, "r") as file:
            all_data = file.readlines()
            if len(all_data) < train_size:
                raise f'train of size of {train_size} selected however only {len(all_data)} datasamples were generated'
            train_samples = random.sample(all_data, train_size)
            val_samples = random.sample(all_data, val_size)
            with open(train_path, 'w') as train, open(val_path, 'w') as val:
                train.writelines(train_samples)
                val.writelines(val_samples)
            create_bk(train_path, out_path, train_size, noise)
            if model == 'popper':
                ##################################
                # out_path = 'output/ilp/datasets/numerical/MichalskiTrains100_0noise/cv_0'
                theory, stats = self.popper_train(out_path, train_log=train_log)
            elif model == 'aleph':
                theory, stats = self.aleph_train(out_path, print_stats=train_log)
            else:
                raise ValueError(f'model: {model} not supported')
            if stats is None:
                print('popper run aborted. no valid theory returned')
            else:
                log_stats(stats, theory, model, train_size, class_rule, noise, 'trainer')

    # def popper_wrapper(self, popper_data, train_log, return_dict):
    #     from popper.loop import learn_solution
    #     from popper.util import Settings, format_prog, order_prog
    #     settings = Settings(popper_data, debug=False, show_stats=train_log, quiet=not train_log, timeout=1000)
    #     prog, score, stats = learn_solution(settings)
    #     # deletes popper prolog files necessary when not
    #     # from pyswip import Prolog
    #     # prolog = Prolog()
    #     # # prolog.consult(os.path.abspath(os.path.abspath('ilp/unloader.pl')))
    #     # for source in list(prolog.query(f'source_file(X).')):
    #     #     path = source['X']
    #     #     # if 'popper' in path:
    #     #         # list(prolog.query(f'unload_source(\'{path}\').'))
    #     #     list(prolog.query(f'unload_file(\'{path}\').'))
    #     theory = None if prog is None else format_prog(order_prog(prog))
    #     return_dict[popper_data] = theory

    def popper_train(self, path, train_log=False):
        from popper.loop import learn_solution
        from popper.util import Settings, format_prog, order_prog
        # print(f'training iteration: {path.split("/")[-1]}')
        popper_data = f'{path}/popper/gt1'
        train_path = f'{path}/train_samples.txt'
        val_path = f'{path}/val_samples.txt'
        settings = Settings(popper_data, debug=False, show_stats=train_log, quiet=(not train_log),
                            timeout=60 * 60 * 4, )

        prog, score, stats = learn_solution(settings)

        # raise AssertionError('Popper run aborted. No valid theory returned.')
        # if train_log:
        #     from popper.util import print_prog_score
        #     print_prog_score(prog, score)
        unload_prolog()
        theory = None if prog is None else format_prog(order_prog(prog))
        if theory is not None:
            stats = eval_rule(theory=theory, ds_val=val_path, ds_train=train_path, dir='TrainGenerator/',
                              print_stats=False, )
            # if queue is not None:
            #     queue.put((theory, stats))
        else:
            print('Popper run aborted. No valid theory returned.')
            return None, None
        return theory, stats

    # @profile
    def aleph_train(self, path, noisy_samples=0, print_stats=False, timeout=60 * 60 * 24):
        aleph_path = f"{path}/aleph"
        train_path = f'{path}/train_samples.txt'
        val_path = f'{path}/val_samples.txt'
        aleph = Aleph()
        aleph.settingsAsFacts(
            'set(i,2), set(clauselength,10), set(minacc,0.7), set(minscore,3), set(minpos,3),'
            ' set(nodes,5000), set(explore,true), set(max_features, 10)')
        aleph.set('minacc', 0.7)
        # enable noise handling, did not work well thus we disabled it
        # if noisy_samples > 0:
        #     aleph.set('noise', noisy_samples)
        # for s in []
        with open(aleph_path + '/trains2/train.b') as background, open(
                aleph_path + '/trains2/train.n') as negative, open(aleph_path + '/trains2/train.f') as pos:
            try:
                theory, features = aleph.induce('induce', pos.read(), negative.read(), background.read(),
                                                printOutput=print_stats, timeout=timeout
                                                )
            except TimeoutError as e:
                warnings.warn(f'Aleph run aborted. Timeout error: {e}')
                return None, None

        # theory = "\n".join([el + '.' for el in t if 'eastbound' in el])
        if theory is not None:
            theory = theory.replace('\n', '').replace('.', '.\n')
            stats = eval_rule(theory=theory, ds_val=val_path, ds_train=train_path, dir='TrainGenerator/',
                              print_stats=False)
        else:
            raise AssertionError('Aleph run aborted. No valid theory returned.')
        del aleph
        return theory, stats

    @staticmethod
    def plot_ilp_crossval(noise=0):
        visualization.plot_ilp_crossval(noise=noise)

    @staticmethod
    def plot_noise_robustness():
        visualization.plot_noise_robustness()

    def to_df(self, out):
        results = pd.DataFrame(
            columns=['Methods', 'training samples', 'rule', 'cv iteration', 'Validation acc', 'theory', 'noise'])
        for p_id, (theory, stats) in enumerate(out):
            if theory is not None:
                li = []
                TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = stats
                log_stats(stats, theory, self.model, self.num_tsamples, self.rule, self.noise, p_id)
                li.append([self.model, self.num_tsamples, self.rule, p_id, (TP + TN) / (TP + FN + TN + FP),
                           (TP_train + TN_train) / (TP_train + FN_train + TN_train + FP_train), theory, self.noise])
                _df = pd.DataFrame(li, columns=['Methods', 'training samples', 'rule', 'cv iteration',
                                                'Validation acc', 'Train acc', 'theory', 'noise'])
                results = pd.concat([results, _df], ignore_index=True)
        return results


def log_stats(stats, theory, model, train_samples, rule, noise, cv_it):
    TP, FN, TN, FP, TP_train, FN_train, TN_train, FP_train = stats
    print(f'#################################################')
    print(f'model: {model}, train samples: {train_samples}, decision rule: {rule} rule, noise: {noise}, cv it: {cv_it}')
    print(theory)
    # print(features)
    print(f'training: ACC:{(TP_train + TN_train) / (FN_train + TN_train + TP_train + FP_train)}, '
          f'Precision:{(TP_train / (TP_train + FP_train)) if (TP_train + FP_train) > 0 else 0}, '
          f'Recall:{(TP_train / (TP_train + FN_train)) if (TP_train + FN_train) > 0 else 0}, '
          f'TP:{TP_train}, FN:{FN_train}, TN:{TN_train}, FP:{FP_train}')
    print(f'Validation: ACC:{(TP + TN) / (FN + TN + TP + FP)}, '
          f'Precision:{(TP / (TP + FP)) if (TP + FP) > 0 else 0}, '
          f'Recall:{(TP / (TP + FN)) if (TP + FN) > 0 else 0}, '
          f'TP:{TP}, FN:{FN}, TN:{TN}, FP:{FP}')


def unload_prolog():
    from pyswip import Prolog
    prolog = Prolog()
    # prolog.consult(os.path.abspath(os.path.abspath('ilp/unloader.pl')))
    for source in list(prolog.query(f'source_file(X).')):
        path = source['X']
        # if 'popper' in path:
        # list(prolog.query(f'unload_source(\'{path}\').'))
        list(prolog.query(f'unload_file(\'{path}\').'))
