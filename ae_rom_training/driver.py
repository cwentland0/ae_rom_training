import os
from functools import partial
import argparse
import pickle

import numpy as np
from hyperopt import fmin, Trials, space_eval

from ae_rom_training.constants import RANDOM_SEED
from ae_rom_training.preproc_utils import read_input_file, get_train_val_data
from ae_rom_training.ml_library import get_ml_library
from ae_rom_training.ml_model.autoencoder import Autoencoder

np.random.seed(RANDOM_SEED)  # seed NumPy RNG


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
    data_train_list, data_val_list = get_train_val_data(input_dict)

    # initialize all autoencoders
    autoencoder_list = []
    for net_idx in range(input_dict["num_networks"]):
        net_suff = input_dict["network_suffixes"][net_idx]
        autoencoder_list.append(Autoencoder(input_dict, mllib, net_suff))

    # optimize each model
    for net_idx in range(input_dict["num_networks"]):

        net_suff = input_dict["network_suffixes"][net_idx]
        data_train = data_train_list[net_idx]
        data_val = data_val_list[net_idx]
        autoencoder = autoencoder_list[net_idx]

        if input_dict["use_hyperopt"]:
            print("Performing hyper-parameter optimization!")
            trials = Trials()
            # wrap objective function to pass additional arguments
            objective_func_wrapped = partial(
                autoencoder.build_and_train,
                input_dict=input_dict,
                data_train=data_train,
                data_val=data_val,
                network_suffix=net_suff,
                mllib=mllib,
            )

            # find "best" model according to specified hyperparameter optimization algorithm
            best = fmin(
                fn=objective_func_wrapped,
                space=autoencoder.param_space,
                algo=input_dict["hyperopt_algo"],
                max_evals=input_dict["hyperopt_max_evals"],
                show_progressbar=False,
                rstate=np.random.RandomState(RANDOM_SEED),
                trials=trials,
            )

            # TODO: train the model again on the full dataset with the best hyper-parameters

            # save HyperOpt metadata to disk
            best_space = space_eval(space, best)
            print("Best parameters:")
            print(best_space)
            f = open(os.path.join(model_dir, "hyperOptTrials" + net_suff + ".pickle"), "wb")
            pickle.dump(trials, f)
            f.close()

        else:
            print("Optimizing single architecture!")
            best = objective_func(space, data_input_train, data_input_val, data_format, model_dir, net_suff, mllib)
            best_space = space

        # write parameter space to file
        f = open(os.path.join(model_dir, "best_space" + net_suff + ".pickle"), "wb")
        pickle.dump(best_space, f)
        f.close()

        # generate explicit batch networks for TensorRT
        # if output_trt:

        #     # load best model
        #     encoder = load_model(os.path.join(model_dir, "encoder" + net_suff + ".h5"), compile=False)
        #     decoder = load_model(os.path.join(model_dir, "decoder" + net_suff + ".h5"), compile=False)
        #     inputShape = encoder.layers[0].input_shape[0][1:]
        #     spatial_dims = data_input_train[0].ndim - 2

        #     # save batch size one network
        #     model_batch_size_one = build_model(best_space, inputShape, spatial_dims, data_format, 1)
        #     decoder_batch_size_one = model_batch_size_one.layers[-1]
        #     encoder_batch_size_one = model_batch_size_one.layers[-2]
        #     decoder_batch_size_one = transfer_weights(decoder, decoder_batch_size_one)
        #     encoder_batch_size_one = transfer_weights(encoder, encoder_batch_size_one)
        #     decoder_batch_size_one.save(os.path.join(model_dir, "decoder_batchOne" + net_suff + ".h5"))
        #     encoder_batch_size_one.save(os.path.join(model_dir, "encoder_batchOne" + net_suff + ".h5"))

        #     # save decoder with batch size equal to latent dimension
        #     model_batch_jacob_decode = build_model(
        #         best_space, inputShape, spatial_dims, data_format, best_space["latent_dim"]
        #     )
        #     decoder_batch_jacob_decode = model_batch_jacob_decode.layers[-1]
        #     decoder_batch_jacob_decode = transfer_weights(decoder, decoder_batch_jacob_decode)
        #     decoder_batch_jacob_decode.save(os.path.join(model_dir, "decoder_batchJacob" + net_suff + ".h5"))

if __name__ == "__main__":
    main()