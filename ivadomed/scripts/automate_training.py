#!/usr/bin/env python
"""
This script enables training and comparison of models on multiple GPUs.

Usage:

```
python scripts/automate_training.py -c path/to/config.json -p path/to/config_hyper.json \
-n number_of_iterations --all-combin
```

"""

import argparse
import copy
import itertools
from functools import partial
import json
import logging
import os
import random
import collections.abc
import shutil
import sys
import joblib
import pandas as pd
import numpy as np
import torch.multiprocessing as mp
from ivadomed import main as ivado
from ivadomed import config_manager as imed_config_manager
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.scripts.compare_models import compute_statistics
from ivadomed import utils as imed_utils

LOG_FILENAME = 'log.txt'
logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, help="Base config file path.",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-p", "--params", required=True,
                        help="JSON file where hyperparameters to experiment are listed.",
                        metavar=imed_utils.Metavar.file)
    parser.add_argument("-n", "--n-iterations", dest="n_iterations", default=1,
                        type=int, help="Number of times to run each config.",
                        metavar=imed_utils.Metavar.int)
    parser.add_argument("--all-combin", dest='all_combin', action='store_true',
                        help="To run all combinations of config"),
    parser.add_argument("-m", "--multi-params", dest="multi_params", action='store_true',
                        help="To change multiple parameters at once.")
    parser.add_argument("--run-test", dest='run_test', action='store_true',
                        help="Evaluate the trained model on the testing sub-set.")
    parser.add_argument("--fixed-split", dest='fixed_split', action='store_true',
                        help="Keep a constant dataset split for all configs and iterations")
    parser.add_argument("-l", "--all-logs", dest="all_logs", action='store_true',
                        help="Keep all log directories for each iteration.")
    parser.add_argument('-t', '--thr-increment', dest="thr_increment", required=False, type=float,
                        help="""A threshold analysis is performed at the end of the training using
                                the trained model and the validation sub-dataset to find the optimal
                                binarization threshold. The specified value indicates the increment
                                between 0 and 1 used during the analysis (e.g. 0.1).""",
                        metavar=imed_utils.Metavar.float)
    parser.add_argument("-o", "--output_dir", required=False,
                        help="Output Folder.")

    return parser


def train_worker(config, thr_incr):
    """
    Args:
        config (dict): dictionary containing configuration details.
        thr_incr (float): A threshold analysis is performed at the end of the training
            using the trained model and the validation sub-dataset to find the optimal binarization
            threshold. The specified value indicates the increment between 0 and 1 used during the
            ROC analysis (e.g. 0.1). Flag: ``-t``, ``--thr-increment``
    """
    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu_ids"] = [config["gpu_ids"][ID]]

    # Call ivado cmd_train
    try:
        # Save best validation score
        best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = \
            ivado.run_command(config, thr_increment=thr_incr)

    except Exception:
        logging.exception('Got exception on main handler')
        logging.info("Unexpected error:", sys.exc_info()[0])
        raise

    # Save config file in log directory
    config_copy = open(config["log_directory"] + "/config_file.json", "w")
    json.dump(config, config_copy, indent=4)

    return config["log_directory"], best_training_dice, best_training_loss, best_validation_dice, \
        best_validation_loss


def test_worker(config):
    # Call ivado cmd_eval

    current = mp.current_process()
    # ID of process used to assign a GPU
    ID = int(current.name[-1]) - 1

    # Use GPU i from the array specified in the config file
    config["gpu_ids"] = [config["gpu_ids"][ID]]

    try:
        # Save best test score
        config["command"] = "test"
        df_results, test_dice = ivado.run_command(config)

    except Exception:
        logging.exception('Got exception on main handler')
        logging.info("Unexpected error:", sys.exc_info()[0])
        raise

    return config["log_directory"], test_dice, df_results


def split_dataset(initial_config):
    train_lst, valid_lst, test_lst = imed_loader_utils.get_new_subject_split(
        path_folder=initial_config["loader_parameters"]["bids_path"],
        center_test=initial_config["split_dataset"]["center_test"],
        split_method=initial_config["split_dataset"]["method"],
        random_seed=initial_config["split_dataset"]["random_seed"],
        train_frac=initial_config["split_dataset"]["train_fraction"],
        test_frac=initial_config["split_dataset"]["test_fraction"],
        log_directory="./",
        balance=initial_config["split_dataset"]['balance'] \
        if 'balance' in initial_config["split_dataset"] else None
    )

    # save the subject distribution
    split_dct = {'train': train_lst, 'valid': valid_lst, 'test': test_lst}
    split_path = "./" + "common_split_datasets.joblib"
    joblib.dump(split_dct, split_path)
    initial_config["split_dataset"]["fname_split"] = split_path
    return initial_config


def make_config_list(param_list, initial_config, all_combin, multi_params):
    """Create a list of config dictionaries corresponding to different hyperparameters.

    Args:
        param_list (list)(HyperparameterOption): A list of the different hyperparameter options.
        initial_config (dict): The original config file, which we use as a basis from which
            to modify our hyperparameters.

            .. code-block:: JSON

                {
                    "training_parameters": {
                        "batch_size": 18,
                        "loss": {"name": "DiceLoss"},
                        "scheduler": {
                            "initial_lr": 0.001
                        }
                    },
                    "default_model":     {
                        "name": "Unet",
                        "dropout_rate": 0.3,
                        "bn_momentum": 0.9,
                        "depth": 3,
                        "is_2d": true
                    },
                    "log_directory": "./tmp/"
                }
        all_combin (bool): If true, combine the hyperparameters combinatorically.
        multi_params (bool): If true, combine the hyperparameters by index in the list, i.e.
            all the first elements, then all the second elements, etc.

    """
    config_list = []
    if all_combin:
        keys = set([hyper_option.base_key for hyper_option in param_list])
        for combination in list(itertools.combinations(param_list, len(keys))):
            if keys_are_unique(combination):
                new_config = copy.deepcopy(initial_config)
                log_dir = new_config['log_directory']
                for hyper_option in combination:
                    new_config = update_dict(new_config, hyper_option.option, hyper_option.base_key)
                    log_dir = log_dir + hyper_option.name
                new_config['log_directory'] = log_dir
                config_list.append(new_config)
    elif multi_params:
        base_keys = get_base_keys(param_list)
        base_key_dict = {key: [] for key in base_keys}
        for hyper_option in param_list:
            base_key_dict[hyper_option.base_key].append(hyper_option)
        max_length = np.min([len(base_key_dict[base_key]) for base_key in base_key_dict.keys()])
        for i in range(0, max_length):
            new_config = copy.deepcopy(initial_config)
            log_dir = new_config['log_directory']
            for key in base_key_dict.keys():
                hyper_option = base_key_dict[key][i]
                new_config = update_dict(new_config, hyper_option.option, hyper_option.base_key)
                log_dir = log_dir + hyper_option.name
            new_config['log_directory'] = log_dir
            config_list.append(new_config)
    else:
        for hyper_option in param_list:
            new_config = copy.deepcopy(initial_config)
            update_dict(new_config, hyper_option.option, hyper_option.base_key)
            new_config['log_directory'] = initial_config["log_directory"] + hyper_option.name
            config_list.append(new_config)

    return config_list


class HyperparameterOption:
    def __init__(self, base_key=None, option=None, base_option=None):
        self.base_key = base_key
        self.option = option
        self.base_option = base_option
        self.name = None
        self.create_name_str()

    def __eq__(self, other):
        return self.base_key == other.base_key and self.option == other.option

    def create_name_str(self):
        self.name = "-" + str(self.base_key) + "=" + str(self.base_option).replace("/", "_")


def get_param_list(my_dict, param_list, superkeys):
    for key, value in my_dict.items():
        if type(value) is list:
            for element in value:
                dict_prev = {key: element}
                for superkey in reversed(superkeys):
                    dict_new = {}
                    dict_new[superkey] = dict_prev
                if len(superkeys) == 0:
                    dict_new = dict_prev
                hyper_option = HyperparameterOption(base_key=key, option=dict_new,
                                                    base_option=element)
                param_list.append(hyper_option)
        else:
            param_list = get_param_list(value, param_list, superkeys + [key])
    return param_list


def update_dict(d, u, base_key):
    for k, v in u.items():
        if k == base_key:
            d[k] = v
        elif isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v, base_key)
        else:
            d[k] = v
    return d


def keys_are_unique(combination):
    keys = [item.base_key for item in combination]
    keys = set(keys)
    return len(keys) == len(combination)


def get_base_keys(param_list):
    base_keys_all = [hyper_option.base_key for hyper_option in param_list]
    base_keys = []
    for base_key in base_keys_all:
        if base_key not in base_keys:
            base_keys.append(base_key)
    return base_keys


def format_results(results_df, config_list, param_list):
    """Merge config and results in a df."""

    config_df = pd.DataFrame.from_dict(config_list)
    print(config_df)
    keep = list(set([list(hyper_option.option.keys())[0] for hyper_option in param_list]))
    keep.append("log_directory")
    config_df = config_df[keep]

    results_df = config_df.set_index('log_directory').join(results_df.set_index('log_directory'))
    results_df = results_df.reset_index()
    results_df = results_df.sort_values(by=['best_validation_loss'])
    return results_df


def automate_training(file_config, file_config_hyper, fixed_split, all_combin, n_iterations=1,
                      run_test=False, all_logs=False, thr_increment=None, multi_params=False,
                      output_dir=None):
    """Automate multiple training processes on multiple GPUs.

    Hyperparameter optimization of models is tedious and time-consuming. This function automatizes
    this optimization across multiple GPUs. It runs trainings, on the same training and validation
    datasets, by combining a given set of parameters and set of values for each of these parameters.
    Results are collected for each combination and reported into a dataframe to allow their
    comparison. The script efficiently allocates each training to one of the available GPUs.

    Usage example::

        ivadomed_automate_training -c config.json -p config_hyper.json -n n_iterations

    .. csv-table:: Example of dataframe
       :file: ../../images/detailed_results.csv

    Config File:
        The config file is the standard config file used in ``ivadomed`` functions. We use this
        as the basis. We call a key of this config file a ``category``. In the example below,
        we would say that ``training_parameters``, ``default_model``, and ``log_directory`` are
        ``categories``.

        .. code-block:: JSON

            {
                "training_parameters": {
                    "batch_size": 18,
                    "loss": {"name": "DiceLoss"},
                    "scheduler": {
                        "initial_lr": 0.001
                    }
                },
                "default_model":     {
                    "name": "Unet",
                    "dropout_rate": 0.3,
                    "depth": 3
                },
                "log_directory": "./tmp/"
            }

    Hyperparameter Config File:
        The hyperparameter config file should have the same layout as the config file. To select
        a hyperparameter you would like to vary, just list the different options under the
        appropriate key, which we call the ``base_key``. In the example below, we want to vary the
        ``batch_size``, ``loss``, and ``depth``; these are our 3 ``base_keys``. As you can see,
        we have listed our different options for these keys. For ``batch_size``, we have listed
        ``2`` and ``64`` as our different options.
        How we implement this depends on 3 settings: ``all_combin``, ``multi_param``,
        or the default.

        .. code-block:: JSON

            {
              "training_parameters": {
                "batch_size": [2, 64],
                "loss": [
                  {"name": "DiceLoss"},
                  {"name": "FocalLoss", "params": {"gamma": 0.2, "alpha" : 0.5}}
                ],
              },
              "default_model": {"depth": [2, 3, 4]}
            }

    Default:
        The default option is to change only one parameter at a time relative to the base
        config file. We then create a list of config options, called ``config_list``.
        Using the examples above, we would have 2 + 2 + 3 = 7 different config options:

        .. code-block:: python

            config_list = [
                {
                    "training_parameters": {
                        "batch_size": 2,
                        "loss": {"name": "DiceLoss"},
                        "scheduler": {
                            "initial_lr": 0.001
                        }
                    },
                    "default_model":     {
                        "name": "Unet",
                        "dropout_rate": 0.3,
                        "depth": 3
                    },
                    "log_directory": "./tmp/"
                },
                {
                    "training_parameters": {
                        "batch_size": 64,
                        "loss": {"name": "DiceLoss"},
                        "scheduler": {
                            "initial_lr": 0.001
                        }
                    },
                    "default_model":     {
                        "name": "Unet",
                        "dropout_rate": 0.3,
                        "depth": 3
                    },
                    "log_directory": "./tmp/"
                },
                {
                    "training_parameters": {
                        "batch_size": 18,
                        "loss": {"name": "DiceLoss"},
                        "scheduler": {
                            "initial_lr": 0.001
                        }
                    },
                    "default_model":     {
                        "name": "Unet",
                        "dropout_rate": 0.3,
                        "depth": 3
                    },
                    "log_directory": "./tmp/"
                },
                # etc ...
            ]


    All Combinations:
        If we select the ``all_combin`` option, we will create a list of configuration options
        combinatorically. Using the config examples above, we would have 2 * 2 * 3 = 12 different
        config options. I'm not going to write out the whole ``config_list`` because it's
        quite long, but here are the combinations:

        .. code-block::

            batch_size = 2, loss = DiceLoss, depth = 2
            batch_size = 2, loss = FocalLoss, depth = 2
            batch_size = 2, loss = DiceLoss, depth = 3
            batch_size = 2, loss = FocalLoss, depth = 3
            batch_size = 2, loss = DiceLoss, depth = 4
            batch_size = 2, loss = FocalLoss, depth = 4
            batch_size = 64, loss = DiceLoss, depth = 2
            batch_size = 64, loss = FocalLoss, depth = 2
            batch_size = 64, loss = DiceLoss, depth = 3
            batch_size = 64, loss = FocalLoss, depth = 3
            batch_size = 64, loss = DiceLoss, depth = 4
            batch_size = 64, loss = FocalLoss, depth = 4

    Multiple Parameters:
        The ``multi_params`` option entails changing all the first elements from the list,
        then all the second parameters from the list, etc. This requires the parameter lists
        within a ``category`` to be the same length. In our example above:

        .. code-block::

            batch_size = 2, loss = DiceLoss, depth = 2
            batch_size = 64, loss = FocalLoss, depth = 3


    Args:
        file_config (string): Configuration filename, which is used as skeleton to configure the
            training. This is the standard config file used in ``ivadomed`` functions. In the
            code, we call the keys from this config file ``categories``.
            Flag: ``--config``, ``-c``
        file_config_hyper (string): json file containing parameters configurations to compare.
            Parameter "keys" of this file need to match the parameter "keys" of `config` file.
            Parameter "values" are in a list. Flag: ``--param``, ``-p``

            Example::

                {"default_model": {"depth": [2, 3, 4]}}

        fixed_split (bool): If True, all the experiments are run on the same
            training/validation/testing subdatasets. Flag: ``--fixed-split``
        all_combin (bool): If True, all parameters combinations are run. Flag: ``--all-combin``
        n_iterations (int): Controls the number of time that each experiment (ie set of parameter)
            are run. Flag: ``--n-iteration``, ``-n``
        run_test (bool): If True, the trained model is also run on the testing subdataset.
            Flag: ``--run-test``
        all_logs (bool): If True, all the log directories are kept for every iteration.
            Flag: ``--all-logs``, ``-l``
        thr_increment (float): A threshold analysis is performed at the end of the training
            using the trained model and the validation sub-dataset to find the optimal binarization
            threshold. The specified value indicates the increment between 0 and 1 used during the
            ROC analysis (e.g. 0.1). Flag: ``-t``, ``--thr-increment``
        multi_params (bool): If True, more than one parameter will be change at the time from
            the hyperparameters. All the first elements from the hyperparameters list will be
            applied, then all the second, etc.
        output_dir (str): Path to where the results will be saved.
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not output_dir:
        output_dir = ""

    # Load initial config
    initial_config = imed_config_manager.ConfigurationManager(file_config).get_config()

    # Split dataset if not already done
    if fixed_split and (initial_config.get("split_path") is None):
        initial_config = split_dataset(initial_config)

    # Hyperparameters values to experiment
    with open(file_config_hyper, "r") as fhandle:
        config_hyper = json.load(fhandle)

    param_list = get_param_list(config_hyper, [], [])
    config_list = make_config_list(param_list, initial_config, all_combin, multi_params)

    # CUDA problem when forking process
    # https://github.com/pytorch/pytorch/issues/2517
    ctx = mp.get_context("spawn")

    # Run all configs on a separate process, with a maximum of n_gpus  processes at a given time
    logging.info(initial_config['gpu_ids'])

    results_df = pd.DataFrame()
    eval_df = pd.DataFrame()
    all_mean = pd.DataFrame()

    with ctx.Pool(processes=len(initial_config["gpu_ids"])) as pool:
        for i in range(n_iterations):
            if not fixed_split:
                # Set seed for iteration
                seed = random.randint(1, 10001)
                for config in config_list:
                    config["split_dataset"]["random_seed"] = seed
                    if all_logs:
                        if i:
                            config["log_directory"] = config["log_directory"].replace("_n=" + str(i - 1).zfill(2),
                                                                                      "_n=" + str(i).zfill(2))
                        else:
                            config["log_directory"] += "_n=" + str(i).zfill(2)

                validation_scores = pool.map(partial(train_worker, thr_incr=thr_increment), config_list)

            val_df = pd.DataFrame(validation_scores, columns=[
                'log_directory', 'best_training_dice', 'best_training_loss', 'best_validation_dice',
                'best_validation_loss'])

            if run_test:
                new_config_list = []
                for config in config_list:
                    # Delete path_pred
                    path_pred = os.path.join(config['log_directory'], 'pred_masks')
                    if os.path.isdir(path_pred) and n_iterations > 1:
                        try:
                            shutil.rmtree(path_pred)
                        except OSError as e:
                            logging.info("Error: %s - %s." % (e.filename, e.strerror))

                    # Take the config file within the log_directory because binarize_prediction may have been updated
                    json_path = os.path.join(config['log_directory'], 'config_file.json')
                    new_config = imed_config_manager.ConfigurationManager(json_path).get_config()
                    new_config["gpu_ids"] = config["gpu_ids"]
                    new_config_list.append(new_config)

                test_results = pool.map(test_worker, new_config_list)

                df_lst = []
                # Merge all eval df together to have a single excel file
                for j, result in enumerate(test_results):
                    df = result[-1]

                    if i == 0:
                        all_mean = df.mean(axis=0)
                        std_metrics = df.std(axis=0)
                        metrics = pd.concat([all_mean, std_metrics], sort=False, axis=1)
                    else:
                        all_mean = pd.concat([all_mean, df.mean(axis=0)], sort=False, axis=1)
                        mean_metrics = all_mean.mean(axis=1)
                        std_metrics = all_mean.std(axis=1)
                        metrics = pd.concat([mean_metrics, std_metrics], sort=False, axis=1)

                    metrics.rename({0: "mean"}, axis=1, inplace=True)
                    metrics.rename({1: "std"}, axis=1, inplace=True)
                    id = result[0].split("_n=")[0]
                    cols = metrics.columns.values
                    for idx, col in enumerate(cols):
                        metrics.rename({col: col + "_" + id}, axis=1, inplace=True)
                    df_lst.append(metrics)
                    test_results[j] = result[:2]

                # Init or add eval results to dataframe
                eval_df = pd.concat(df_lst, sort=False, axis=1)

                test_df = pd.DataFrame(test_results, columns=['log_directory', 'test_dice'])
                combined_df = val_df.set_index('log_directory').join(test_df.set_index('log_directory'))
                combined_df = combined_df.reset_index()

            else:
                combined_df = val_df

            results_df = pd.concat([results_df, combined_df])
            results_df.to_csv(os.path.join(output_dir, "temporary_results.csv"))
            eval_df.to_csv(os.path.join(output_dir, "average_eval.csv"))

    results_df = format_results(results_df, config_list, param_list)
    results_df.to_csv(os.path.join(output_dir, "detailed_results.csv"))

    logging.info("Detailed results")
    logging.info(results_df)

    # Compute avg, std, p-values
    if n_iterations > 1:
        compute_statistics(results_df, n_iterations, run_test)


def main(args=None):
    imed_utils.init_ivadomed()
    parser = get_parser()
    args = imed_utils.get_arguments(parser, args)

    thr_increment = args.thr_increment if args.thr_increment else None

    automate_training(file_config=args.config,
                      file_config_hyper=args.params,
                      fixed_split=bool(args.fixed_split),
                      all_combin=bool(args.all_combin),
                      n_iterations=int(args.n_iterations),
                      run_test=bool(args.run_test),
                      all_logs=args.all_logs,
                      thr_increment=thr_increment,
                      multi_params=bool(args.multi_params),
                      output_dir=args.output_dir
                      )


if __name__ == '__main__':
    main()
