import numpy as np
import os
from param_defs import define_param_space
from train_autoencoder import objective_func, build_model
from preproc_utils import agg_data_sets
from cnn_utils import transfer_weights
from misc_utils import get_vars_from_data, read_input_file, catch_input
from hyperopt import fmin, tpe, rand, Trials, space_eval
from functools import partial
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.random import set_seed
import argparse
import pickle

np.random.seed(24)  # seed NumPy RNG
set_seed(24)

# ----- START USER INPUTS -----

# read working directory input
parser = argparse.ArgumentParser(description="Read input file")
parser.add_argument("input_file", type=str, help="input file")
input_file = os.path.expanduser(parser.parse_args().input_file)
assert os.path.isfile(input_file), "Given input_file does not exist"
input_dict = read_input_file(input_file)

run_cpu = catch_input(input_dict, "run_cpu", False)

data_dir = input_dict["data_dir"]
model_dir = input_dict["model_dir"]
model_label = input_dict["model_label"]
data_files_train = input_dict["data_files_train"]
data_files_val = catch_input(input_dict, "data_files_val", [None])
var_network_idxs = list(input_dict["var_network_idxs"])

num_networks = len(var_network_idxs)
idx_start_list = catch_input(input_dict, "idx_start_list", [0] * num_networks)
idx_end_list = catch_input(input_dict, "idx_end_list", [None] * num_networks)
idx_skip_list = catch_input(input_dict, "idx_skip_list", [None] * num_networks)
data_order = input_dict["data_order"]
network_order = input_dict["network_order"]

# HyperOpt parameters
use_hyper_opt = catch_input(input_dict, "use_hyper_opt", False)
hyper_opt_algo = catch_input(input_dict, "hyper_opt_algo", "tpe")
hyper_opt_max_evals = catch_input(input_dict, "hyper_opt_max_evals", 100)

# for TensorRT compliance
output_trt = catch_input(input_dict, "output_trt", False)

# ----- END USER INPUTS -----

# run on CPU vs GPU
if run_cpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
else:
    # make sure TF doesn't gobble up device memory
    gpus = tf.config.experimental.list_physical_devices("GPU")
    if gpus:
        try:
            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                logical_gpus = tf.config.experimental.list_logical_devices("GPU")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

# TODO: batch norm back into network
# TODO: add option for all-convolution network

model_dir = os.path.join(model_dir, model_label)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# setting preprocessing
space = define_param_space(input_dict, use_hyper_opt)

network_suffixes = [""] * num_networks
for i, net_idxs in enumerate(var_network_idxs):
    for j, idx in enumerate(net_idxs):
        network_suffixes[i] += "_" + str(idx)

# delete loss value file, so run overwrites anything already in folder
for network_suffix in network_suffixes:
    loss_loc = os.path.join(model_dir, "valLoss" + network_suffix + ".dat")
    try:
        os.remove(loss_loc)
    except FileNotFoundError:
        pass

# hyperOpt search algorithm
# 'rand' for random search, 'tpe' for tree-structured Parzen estimator
if use_hyper_opt:
    if hyper_opt_algo == "rand":
        hyper_opt_algo = rand.suggest
    elif hyper_opt_algo == "tpe":
        hyper_opt_algo = tpe.suggest
    else:
        raise ValueError("Invalid input for hyper_opt_algo: " + str(hyper_opt_algo))

# ----- LOAD RAW DATA -----

num_datasets_train = len(data_files_train)
if len(idx_start_list) == 1:
    idx_start_list = idx_start_list * num_datasets_train
if len(idx_end_list) == 1:
    idx_end_list = idx_end_list * num_datasets_train
if len(idx_skip_list) == 1:
    idx_skip_list = idx_skip_list * num_datasets_train

data_raw_train = agg_data_sets(data_dir, data_files_train, idx_start_list, idx_end_list, idx_skip_list, data_order)
if data_files_val[0] is not None:
    data_raw_val = agg_data_sets(data_dir, data_files_val, idx_start_list, idx_end_list, idx_skip_list, data_order)
else:
    data_raw_val = None

if network_order == "NCHW":
    data_format = "channels_first"
elif network_order == "NHWC":
    data_format = "channels_last"
else:
    raise ValueError("Invalid network_order: " + str(network_order))

# ----- MODEL OPTIMIZATION -----
# optimize as many models as requested, according to variable split
for net_idx in range(num_networks):

    net_suff = network_suffixes[net_idx]
    if var_network_idxs is None:
        data_input_train = data_raw_train
        data_input_val = data_raw_val
    else:
        data_input_train = get_vars_from_data(data_raw_train, net_idx, var_network_idxs)
        if data_files_val[0] is not None:
            data_input_val = get_vars_from_data(data_raw_val, net_idx, var_network_idxs)
        else:
            data_input_val = data_raw_val

    if use_hyper_opt:
        print("Performing hyper-parameter optimization!")
        trials = Trials()
        # wrap objective function to pass additional arguments
        objective_func_wrapped = partial(
            objective_func,
            dataList_train=data_input_train,
            dataList_val=data_input_val,
            data_format=data_format,
            model_dir=model_dir,
            network_suffix=net_suff,
        )

        # find "best" model according to specified hyperparameter optimization algorithm
        best = fmin(
            fn=objective_func_wrapped,
            space=space,
            algo=hyper_opt_algo,
            max_evals=hyper_opt_max_evals,
            show_progressbar=False,
            rstate=np.random.RandomState(24),
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
        best = objective_func(space, data_input_train, data_input_val, data_format, model_dir, net_suff)
        best_space = space

    # write parameter space to file
    f = open(os.path.join(model_dir, "best_space" + net_suff + ".pickle"), "wb")
    pickle.dump(best_space, f)
    f.close()

    # generate explicit batch networks for TensorRT
    if output_trt:

        # load best model
        encoder = load_model(os.path.join(model_dir, "encoder" + net_suff + ".h5"), compile=False)
        decoder = load_model(os.path.join(model_dir, "decoder" + net_suff + ".h5"), compile=False)
        inputShape = encoder.layers[0].input_shape[0][1:]
        spatial_dims = data_input_train[0].ndim - 2

        # save batch size one network
        model_batch_size_one = build_model(best_space, inputShape, spatial_dims, data_format, 1)
        decoder_batch_size_one = model_batch_size_one.layers[-1]
        encoder_batch_size_one = model_batch_size_one.layers[-2]
        decoder_batch_size_one = transfer_weights(decoder, decoder_batch_size_one)
        encoder_batch_size_one = transfer_weights(encoder, encoder_batch_size_one)
        decoder_batch_size_one.save(os.path.join(model_dir, "decoder_batchOne" + net_suff + ".h5"))
        encoder_batch_size_one.save(os.path.join(model_dir, "encoder_batchOne" + net_suff + ".h5"))

        # save decoder with batch size equal to latent dimension
        model_batch_jacob_decode = build_model(
            best_space, inputShape, spatial_dims, data_format, best_space["latent_dim"]
        )
        decoder_batch_jacob_decode = model_batch_jacob_decode.layers[-1]
        decoder_batch_jacob_decode = transfer_weights(decoder, decoder_batch_jacob_decode)
        decoder_batch_jacob_decode.save(os.path.join(model_dir, "decoder_batchJacob" + net_suff + ".h5"))
