import os
from functools import partial
import argparse
import pickle
from time import time

import numpy as np
from hyperopt import fmin, Trials, space_eval

from ae_rom_training.constants import RANDOM_SEED
from ae_rom_training.preproc_utils import read_input_file, get_train_val_data
from ae_rom_training.ml_library import get_ml_library
from ae_rom_training.ae_rom.baseline_ae_rom import BaselineAEROM
from ae_rom_training.ae_rom.koopman_ae_otto2019 import KoopmanAEOtto2019
from ae_rom_training.ae_rom.koopman_ae_pan2020 import KoopmanAEPan2020

np.random.seed(RANDOM_SEED)  # seed NumPy RNG

# TODO: detect if a component has no Hyperopt expressions, don't use Hyperopt
# TODO: load trained autoencoder for separate training of time-stepper
# TODO: define layer precision
# TODO: dry run option to display ALL layer parameters so user can verify before training


def main():

    # read input file and do some setup
    parser = argparse.ArgumentParser(description="Read input file")
    parser.add_argument("input_file", type=str, help="input file")
    input_file = os.path.expanduser(parser.parse_args().input_file)
    assert os.path.isfile(input_file), "Given input_file does not exist"
    input_dict = read_input_file(input_file)

    # clear out old results
    # TODO: need to finalize whether this is a good thing, do I trust myself to not delete good results accidentally?
    for network_suffix in input_dict["network_suffixes"]:
        loss_loc = os.path.join(input_dict["model_dir"], "val_loss" + network_suffix + ".dat")
        try:
            os.remove(loss_loc)
        except FileNotFoundError:
            pass

    # get ML library to use for this training session
    mllib = get_ml_library(input_dict)

    # get training and validation data
    data_list_train, data_list_val, split_idxs_list_train, split_idxs_list_val = get_train_val_data(input_dict)

    # initialize all autoencoders
    # TODO: move this junk somewhere else
    aerom_list = []
    aerom_type = input_dict["aerom_type"]
    for net_idx in range(input_dict["num_networks"]):
        net_suff = input_dict["network_suffixes"][net_idx]
        if aerom_type == "baseline":
            aerom_list.append(BaselineAEROM(input_dict, mllib, net_suff))
        elif aerom_type == "koopman_otto2019":
            aerom_list.append(KoopmanAEOtto2019(input_dict, mllib, network_suffix))
        elif aerom_type == "koopman_pan2020":
            aerom_list.append(KoopmanAEPan2020(input_dict, mllib, network_suffix))
        else:
            raise ValueError("Invalid aerom_type selection: " + str(aerom_type))

    # optimize each model
    time_start_full = time()
    for net_idx in range(input_dict["num_networks"]):

        print("=================================================================")
        print("TRAINING NETWORK " + str(net_idx + 1) + "/" + str(input_dict["num_networks"]))
        print("=================================================================")
        time_start_network = time()

        net_suff = input_dict["network_suffixes"][net_idx]
        data_list_train_net = data_list_train[net_idx]
        data_list_val_net = data_list_val[net_idx]
        input_dict["split_idxs_train"] = split_idxs_list_train[net_idx]
        input_dict["split_idxs_val"] = split_idxs_list_val[net_idx]

        aerom = aerom_list[net_idx]

        input_dict["trial_number"] = 1

        if input_dict["use_hyperopt"]:

            print("\nPERFORMING HYPER-PARAMETER OPTIMIZATION\n")

            # wrap objective function to pass additional arguments
            if (aerom.time_stepper is None) or (
                (aerom.time_stepper is not None) and (aerom.training_format == "separate")
            ):
                objective_func_wrapped = partial(
                    aerom.build_and_train,
                    input_dict=input_dict,
                    data_list_train=data_list_train_net,
                    data_list_val=data_list_val_net,
                    ae=True,
                )

            elif (aerom.time_stepper is not None) and (aerom.training_format == "combined"):
                objective_func_wrapped = partial(
                    aerom.build_and_train,
                    input_dict=input_dict,
                    data_list_train=data_list_train_net,
                    data_list_val=data_list_val_net,
                    ae=True,
                    ts=True,
                )
            else:
                raise ValueError("Something unexpected happened in selecting build_and_train")

            # find "best" model according to specified hyperparameter optimization algorithm
            trials = Trials()
            best = fmin(
                fn=objective_func_wrapped,
                space=aerom.param_space,
                algo=input_dict["hyperopt_algo"],
                max_evals=input_dict["hyperopt_max_evals"],
                show_progressbar=False,
                rstate=np.random.RandomState(RANDOM_SEED),
                trials=trials,
            )

            # TODO: train the model again on the full dataset with the best hyper-parameters

            # save HyperOpt metadata to disk
            best_space = space_eval(aerom.param_space, best)
            print("Best parameters:")
            print(best_space)
            f = open(os.path.join(input_dict["model_dir"], "hyper_opt_trials" + net_suff + ".pickle"), "wb")
            pickle.dump(trials, f)
            f.close()

        else:
            print("\nTRAINING SINGLE ARCHITECTURE\n")

            # train autoencoder alone
            if (aerom.time_stepper is None) or (
                (aerom.time_stepper is not None) and (aerom.training_format == "separate")
            ):
                best = aerom.build_and_train(
                    aerom.param_space, input_dict, data_list_train_net, data_list_val_net, ae=True
                )

            # train autoencoder and time stepper together
            elif (aerom.time_stepper is not None) and (aerom.training_format == "combined"):
                best = aerom.build_and_train(
                    aerom.param_space, input_dict, data_list_train_net, data_list_val_net, ae=True, ts=True
                )

            else:
                raise ValueError("Something unexpected happened in selecting build_and_train")

            best_space = aerom.param_space

        time_end_network = time()
        print("=================================================================")
        print("NETWORK TRAINING COMPLETE IN " + str(time_end_network - time_start_network) + " seconds")
        print("=================================================================")

        # write parameters to file
        f = open(os.path.join(input_dict["model_dir"], "best_params" + net_suff + ".pickle"), "wb")
        pickle.dump(best_space, f)
        f.close()

    time_end_full = time()
    print("=================================================================")
    print("TOTAL TRAINING COMPLETE IN " + str(time_end_full - time_start_full) + " seconds")
    print("=================================================================")


if __name__ == "__main__":
    main()
