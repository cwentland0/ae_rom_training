import os
from functools import partial
import argparse
import pickle
from time import time

import numpy as np
from hyperopt import fmin, Trials, space_eval

from ae_rom_training.constants import RANDOM_SEED
from ae_rom_training.preproc_utils import catch_input, read_input_file, get_train_val_data, seed_rng
from ae_rom_training.ml_library import get_ml_library
from ae_rom_training.ae_rom.baseline_ae_rom import BaselineAEROM
from ae_rom_training.ae_rom.koopman_ae_discrete import KoopmanAEDiscrete
from ae_rom_training.ae_rom.koopman_ae_continuous import KoopmanAEContinuous
from ae_rom_training.ae_rom.generic_recurrent_ae_ts import GenericRecurrentAETS

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
    model_dir = input_dict["model_dir"]

    # clear out old results
    # TODO: need to finalize whether this is a good thing, do I trust myself to not delete good results accidentally?
    for network_suffix in input_dict["network_suffixes"]:
        for train_prefix in ["", "ae_", "ts_"]:
            loss_loc = os.path.join(model_dir, train_prefix + "val_loss" + network_suffix + ".dat")
            try:
                os.remove(loss_loc)
            except FileNotFoundError:
                pass

    # get ML library to use for this training session
    run_gpu = catch_input(input_dict, "run_gpu", False)
    mllib = get_ml_library(input_dict["mllib_name"], run_gpu)

    # initial RNG seed
    seed_rng(mllib)

    # get training and validation data
    data_list_train, data_list_val, split_idxs_list_train, split_idxs_list_val = get_train_val_data(input_dict)

    # initialize all autoencoders
    # TODO: move this junk somewhere else
    # TODO: just instantiate once then do a deep copy, overwrite network suffix
    aerom_list = []
    aerom_type = input_dict["aerom_type"]
    for net_idx in range(input_dict["num_networks"]):
        net_suff = input_dict["network_suffixes"][net_idx]
        if aerom_type == "baseline_ae":
            aerom_list.append(BaselineAEROM(net_idx, input_dict, mllib, net_suff))
        elif aerom_type == "koopman_discrete":
            aerom_list.append(KoopmanAEDiscrete(net_idx, input_dict, mllib, network_suffix))
        elif aerom_type == "koopman_continuous":
            aerom_list.append(KoopmanAEContinuous(net_idx, input_dict, mllib, network_suffix))
        elif aerom_type == "generic_recurrent":
            aerom_list.append(GenericRecurrentAETS(net_idx, input_dict, mllib, network_suffix))
        else:
            raise ValueError("Invalid aerom_type selection: " + str(aerom_type))

    # optimize each model
    time_start_full = time()
    for net_idx in range(input_dict["num_networks"]):

        print("=================================================================")
        print("TRAINING NETWORK " + str(net_idx + 1) + "/" + str(input_dict["num_networks"]))
        print("=================================================================")
        time_start_network = time()

        # reset RNG
        seed_rng(mllib)

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
            if input_dict["train_ae"] != input_dict["train_ts"]:
                if input_dict["train_ae"]:
                    objective_func_wrapped = partial(
                        aerom.build_and_train,
                        input_dict=input_dict,
                        data_list_train=data_list_train_net,
                        data_list_val=data_list_val_net,
                        training_ae=True,
                        training_ts=False,
                    )
                else:
                    objective_func_wrapped = partial(
                        aerom.build_and_train,
                        input_dict=input_dict,
                        data_list_train=data_list_train_net,
                        data_list_val=data_list_val_net,
                        training_ae=False,
                        training_ts=True,
                    )
            else:
                objective_func_wrapped = partial(
                    aerom.build_and_train,
                    input_dict=input_dict,
                    data_list_train=data_list_train_net,
                    data_list_val=data_list_val_net,
                    training_ae=True,
                    training_ts=True,
                )

            # breakpoint()
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
            f = open(os.path.join(model_dir, "hyper_opt_trials" + net_suff + ".pickle"), "wb")
            pickle.dump(trials, f)
            f.close()

        else:

            print("\nTRAINING SINGLE ARCHITECTURE\n")

            # train either autoencoder or time stepper
            if input_dict["train_ae"] != input_dict["train_ts"]:
                if input_dict["train_ae"]:
                    best = aerom.build_and_train(
                        aerom.param_space, input_dict, data_list_train_net, data_list_val_net, training_ae=True,
                    )
                else:
                    best = aerom.build_and_train(
                        aerom.param_space, input_dict, data_list_train_net, data_list_val_net, training_ts=True
                    )
                write_param_space(aerom.param_space, model_dir, aerom.train_prefix, "best_params", net_suff)

            # train networks together
            else:
                best = aerom.build_and_train(
                    aerom.param_space,
                    input_dict,
                    data_list_train_net,
                    data_list_val_net,
                    training_ae=True,
                    training_ts=True,
                )
                write_param_space(
                    aerom.param_space, model_dir, aerom.train_prefix, "best_params", net_suff,
                )

        time_end_network = time()
        print("=================================================================")
        print("NETWORK TRAINING COMPLETE IN " + str(time_end_network - time_start_network) + " seconds")
        print("=================================================================")

    time_end_full = time()
    print("=================================================================")
    print("TOTAL TRAINING COMPLETE IN " + str(time_end_full - time_start_full) + " seconds")
    print("=================================================================")


def write_param_space(space, model_dir, prefix, space_name, suffix):
    """Pickle and save parameter space"""

    # write parameters to file
    f = open(os.path.join(model_dir, prefix + space_name + suffix + ".pickle"), "wb")
    pickle.dump(space, f)
    f.close()


if __name__ == "__main__":
    main()
