import re
import os

import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import tpe, rand

from ae_rom_training.constants import RANDOM_SEED


def parse_value(expr):
    """
    Parse read text value into dict value
    """

    try:
        return eval(expr)
    except:
        return eval(re.sub("\s+", ",", expr))
    else:
        return expr


def parse_line(line):
    """
    Parse read text line into dict key and value
    """

    eq = line.find("=")
    if eq == -1:
        raise Exception()
    key = line[:eq].strip()
    value = line[eq + 1 : -1].strip()
    return key, parse_value(value)


def read_text_file(inputFile):

    # TODO: better exception handling besides just a pass

    read_dict = {}
    with open(inputFile) as f:
        contents = f.readlines()

    for line in contents:
        try:
            key, val = parse_line(line)
            read_dict[key] = val
            # convert lists to NumPy arrays
            # if (type(val) == list):
            # 	read_dict[key] = np.asarray(val)
        except:
            pass

    return read_dict


def catch_input(in_dict, in_key, default_val):

    default_type = type(default_val)
    try:
        # if NoneType passed as default, trust user
        if isinstance(default_type, type(None)):
            out_val = in_dict[in_key]
        else:
            out_val = default_type(in_dict[in_key])
    except:
        out_val = default_val

    return out_val


def read_input_file(input_file):
    """Read text input file for fixed parameters.

    This reads every input parameter from input_file and does some manual checking for FIXED parameters.
    Network parameters are checked when the autoencoders are initialized.
    Variable training parameters are checked right before training.
    """

    # TODO: lots more error catching

    # raw inputs, no defaults assigned
    input_dict_raw = read_text_file(input_file)

    # this really just checks that all necessary inputs are given and defaults are assigned otherwise
    input_dict = input_dict_raw.copy()

    # path inputs
    input_dict["data_dir"] = input_dict_raw["data_dir"]
    input_dict["data_files_train"] = input_dict_raw["data_files_train"]
    input_dict["data_files_val"] = catch_input(input_dict_raw, "data_files_val", [None])
    input_dict["model_dir"] = os.path.join(input_dict_raw["model_dir"], input_dict_raw["model_label"])
    if not os.path.exists(input_dict["model_dir"]):
        os.makedirs(input_dict["model_dir"])

    # data inputs
    input_dict["var_network_idxs"] = list(input_dict_raw["var_network_idxs"])
    input_dict["num_networks"] = len(input_dict["var_network_idxs"])
    input_dict["idx_start_list_train"] = catch_input(input_dict_raw, "idx_start_list_train", [0])
    input_dict["idx_end_list_train"] = catch_input(input_dict_raw, "idx_end_list_train", [None])
    input_dict["idx_skip_list_train"] = catch_input(input_dict_raw, "idx_skip_list_train", [None])
    input_dict["idx_start_list_val"] = catch_input(input_dict_raw, "idx_start_list_val", [0])
    input_dict["idx_end_list_val"] = catch_input(input_dict_raw, "idx_end_list_val", [None])
    input_dict["idx_skip_list_val"] = catch_input(input_dict_raw, "idx_skip_list_val", [None])
    input_dict["data_order"] = input_dict_raw["data_order"]
    input_dict["network_order"] = catch_input(input_dict_raw, "network_order", "NHWC")
    input_dict["network_suffixes"] = [""] * input_dict["num_networks"]
    for i, net_idxs in enumerate(input_dict["var_network_idxs"]):
        for j, idx in enumerate(net_idxs):
            input_dict["network_suffixes"][i] += "_" + str(idx)
    num_datasets_train = len(input_dict["data_files_train"])
    if len(input_dict["idx_start_list_train"]) == 1:
        input_dict["idx_start_list_train"] *= num_datasets_train
    if len(input_dict["idx_end_list_train"]) == 1:
        input_dict["idx_end_list_train"] *= num_datasets_train
    if len(input_dict["idx_skip_list_train"]) == 1:
        input_dict["idx_skip_list_train"] *= num_datasets_train
    if len(input_dict["idx_start_list_val"]) == 1:
        input_dict["idx_start_list_val"] *= num_datasets_train
    if len(input_dict["idx_end_list_val"]) == 1:
        input_dict["idx_end_list_val"] *= num_datasets_train
    if len(input_dict["idx_skip_list_val"]) == 1:
        input_dict["idx_skip_list_val"] *= num_datasets_train

    # global parameters
    input_dict["aerom_type"] = input_dict_raw["aerom_type"]
    input_dict["split_scheme"] = input_dict_raw["split_scheme"]
    input_dict["centering_scheme"] = input_dict_raw["centering_scheme"]
    input_dict["normal_scheme"] = input_dict_raw["normal_scheme"]
    input_dict["val_perc"] = input_dict_raw["val_perc"]
    input_dict["training_format"] = catch_input(input_dict_raw, "training_format", "separate")
    input_dict["precision"] = catch_input(
        input_dict_raw, "precision", "32"
    )  # string because "mixed" will be an option later
    input_dict["mirrored_decoder"] = catch_input(input_dict_raw, "mirrored_decoder", False)

    # HyperOpt parameters
    input_dict["use_hyperopt"] = catch_input(input_dict_raw, "use_hyperopt", False)
    input_dict["hyperopt_algo"] = catch_input(input_dict_raw, "hyperopt_algo", "tpe")
    input_dict["hyperopt_max_evals"] = catch_input(input_dict_raw, "hyperopt_max_evals", 100)
    if input_dict["use_hyperopt"]:
        if input_dict["hyperopt_algo"] == "rand":
            input_dict["hyperopt_algo"] = rand.suggest
        elif input_dict["hyperopt_algo"] == "tpe":
            input_dict["hyperopt_algo"] = tpe.suggest
        else:
            raise ValueError("Invalid input for hyperopt_algo: " + str(input_dict["hyperopt_algo"]))

    return input_dict


def get_vars_from_data(data_list, var_idxs):
    """Retrieve data subarray for specified variables

    Assumes structured data in NCHW format.
    """

    data_input = []
    num_spatial_dims = data_list[0].ndim - 2
    for data_mat in data_list:
        if num_spatial_dims == 1:
            data_input.append(data_mat[:, var_idxs, :])
        elif num_spatial_dims == 2:
            data_input.append(data_mat[:, var_idxs, :, :])
        else:
            data_input.append(data_mat[:, var_idxs, :, :, :])

    return data_input


def get_train_val_data(input_dict):

    # get training data set
    data_list_raw_train = agg_data_sets(
        input_dict["data_dir"],
        input_dict["data_files_train"],
        input_dict["idx_start_list_train"],
        input_dict["idx_end_list_train"],
        input_dict["idx_skip_list_train"],
        input_dict["data_order"],
    )
    # assumed spatially-oriented data, so subtract samples and channels dimensions
    input_dict["num_dims"] = data_list_raw_train[0].ndim - 2

    # get validation data sets, if given
    if input_dict["data_files_val"][0] is not None:
        data_list_raw_val = agg_data_sets(
            input_dict["data_dir"],
            input_dict["data_files_val"],
            input_dict["idx_start_list_val"],
            input_dict["idx_end_list_val"],
            input_dict["idx_skip_list_val"],
            input_dict["data_order_val"],
        )
    else:
        data_list_raw_val = None

    # TODO: training/validation split is different for each network, might not be the more rigorous?
    # might be easier to get a fixed index shuffle, apply to all network
    # as this stands right now, no real point to doing centering/normalization separately, since there's
    #   no option to do separate centering/normalization/split schemes for different variables
    data_list_train, data_list_val = [], []
    for net_idx, var_idxs in enumerate(input_dict["var_network_idxs"]):

        data_list_var_train = get_vars_from_data(data_list_raw_train, var_idxs)
        if data_list_raw_val is not None:
            data_list_var_val = get_vars_from_data(data_list_raw_val, var_idxs)
        else:
            data_list_var_val = None

        # pre-process data
        # includes centering, normalization, and train/validation split
        data_var_train, data_var_val = preproc_raw_data(
            data_list_var_train,
            input_dict["centering_scheme"],
            input_dict["split_scheme"],
            input_dict["normal_scheme"],
            input_dict["model_dir"],
            input_dict["network_suffixes"][net_idx],
            data_list_val=data_list_var_val,
            val_perc=input_dict["val_perc"],
        )

        # up until now, data has been in NCHW, tranpose if requesting NHWC
        if input_dict["network_order"] == "NHWC":
            if input_dict["num_dims"] == 1:
                trans_axes = (0, 2, 1)
            elif input_dict["num_dims"] == 2:
                trans_axes = (0, 2, 3, 1)
            elif input_dict["num_dims"] == 3:
                trans_axes = (0, 2, 3, 4, 1)
            data_var_train = np.transpose(data_var_train, trans_axes)
            data_var_val = np.transpose(data_var_val, trans_axes)

        data_list_train.append(data_var_train)
        data_list_val.append(data_var_val)

    return data_list_train, data_list_val


def agg_data_sets(data_dir, data_loc_list, idx_start_list, idx_end_list, idx_skip_list, data_order):
    """Given list of data locations, aggregate data sets.

    Puts all data in NCHW format.
    """

    data_raw = []
    for file_count, data_file in enumerate(data_loc_list):
        data_loc = os.path.join(data_dir, data_file)
        data_load = np.load(data_loc)

        num_dims = data_load.ndim - 2  # excludes N and C dimensions

        # For now, everything goes to NCHW, will get transposed to NHWC right before training if requested
        if data_order != "NCHW":
            if data_order == "NHWC":
                if num_dims == 1:
                    trans_axes = (0, 2, 1)
                elif num_dims == 2:
                    trans_axes = (0, 3, 1, 2)
                elif num_dims == 3:
                    trans_axes = (0, 4, 1, 2, 3)

            elif data_order == "HWCN":
                if num_dims == 1:
                    trans_axes = (2, 1, 0)
                elif num_dims == 2:
                    trans_axes = (3, 2, 0, 1)
                elif num_dims == 3:
                    trans_axes = (4, 3, 0, 1, 2)

            elif data_order == "HWNC":
                if num_dims == 1:
                    trans_axes = (1, 2, 0)
                elif num_dims == 2:
                    trans_axes = (2, 3, 0, 1)
                elif num_dims == 3:
                    trans_axes = (3, 4, 0, 1, 2)

            elif data_order == "CHWN":
                if num_dims == 1:
                    trans_axes = (2, 0, 1)
                elif num_dims == 2:
                    trans_axes = (3, 0, 1, 2)
                elif num_dims == 3:
                    trans_axes = (4, 0, 1, 2, 3)

            elif data_order == "CNHW":
                if num_dims == 1:
                    trans_axes = (1, 0, 2)
                elif num_dims == 2:
                    trans_axes = (1, 0, 2, 3)
                elif num_dims == 3:
                    trans_axes = (1, 0, 2, 3, 4)

            else:
                raise ValueError("Invalid data_order: " + str(data_order))

            data_load = np.transpose(data_load, axes=trans_axes)

        # extract a range of iterations
        if num_dims == 1:
            data_load = data_load[
                idx_start_list[file_count] : idx_end_list[file_count] : idx_skip_list[file_count], :, :
            ]
        elif num_dims == 2:
            data_load = data_load[
                idx_start_list[file_count] : idx_end_list[file_count] : idx_skip_list[file_count], :, :, :
            ]
        elif num_dims == 3:
            data_load = data_load[
                idx_start_list[file_count] : idx_end_list[file_count] : idx_skip_list[file_count], :, :, :, :
            ]

        # aggregate all data sets
        data_raw.append(data_load.copy())

    return data_raw


def preproc_raw_data(
    data_list_train,
    centering_scheme,
    split_scheme,
    normal_scheme,
    model_dir,
    network_suffix,
    data_list_val=None,
    val_perc=0.0,
):

    # make train/val split from given training data
    if data_list_val is None:

        # concatenate samples after centering
        for dataset_num, data_arr in enumerate(data_list_train):
            data_in = center_data_set(data_arr, centering_scheme, model_dir, network_suffix, save_cent=True)
            data_in_train, data_in_val = split_data_set(data_in, split_scheme, val_perc)
            if dataset_num == 0:
                data_train = data_in_train.copy()
                data_val = data_in_val.copy()
            else:
                # TODO: this format causes problems for time series split
                data_train = np.append(data_train, data_in_train, axis=0)
                data_val = np.append(data_val, data_in_val, axis=0)

    # training/validation split given by files
    else:
        # aggregate training samples after centering
        for dataset_num, data_arr in enumerate(data_list_train):
            data_in_train = center_data_set(data_arr, centering_scheme, model_dir, network_suffix, save_cent=True)
            if dataset_num == 0:
                data_train = data_in_train.copy()
            else:
                data_train = np.append(data_train, data_in_train, axis=0)
        # shuffle training data to avoid temporal/dataset bias (shuffles along FIRST axis)
        np.random.shuffle(data_train)

        # aggregate validation samples after sampling
        # don't need to shuffle validation data
        for dataset_num, data_arr in enumerate(data_list_val):
            data_in_val = center_data_set(data_arr, centering_scheme, model_dir, network_suffix, save_cent=False)
            if dataset_num == 0:
                data_val = data_in_val.copy()
            else:
                data_val = np.append(data_val, data_in_val, axis=0)

    # normalize training and validation sets separately
    data_train, norm_sub_train, norm_fac_train = normalize_data_set(
        data_train, normal_scheme, model_dir, network_suffix, save_norm=True
    )
    data_val, _, _ = normalize_data_set(
        data_val, normal_scheme, model_dir, network_suffix, norms=[norm_sub_train, norm_fac_train], save_norm=False
    )

    return data_train, data_val


# assumed to be in NCHW format
def center_data_set(data, cent_type, model_dir, network_suffix, save_cent=False):

    num_dims = data.ndim - 2

    if cent_type == "init_cond":
        if num_dims == 1:
            cent_prof = data[[0], :, :]
        elif num_dims == 2:
            cent_prof = data[[0], :, :, :]
        elif num_dims == 3:
            cent_prof = data[[0], :, :, :, :]
        else:
            raise ValueError("Something went wrong with centering (data dimensions)")

    elif cent_type == "none":
        cent_prof = np.zeros((1,) + data.shape[1:], dtype=np.float64)

    else:
        raise ValueError("Invalid choice of cent_type: " + cent_type)

    data = data - cent_prof

    if save_cent:

        cent_prof = np.squeeze(cent_prof, axis=0)
        np.save(os.path.join(model_dir, "cent_prof" + network_suffix + ".npy"), cent_prof)

    return data


def split_data_set(data, split_type, val_perc):

    if split_type == "random":
        data_train, data_val = train_test_split(data, test_size=val_perc, random_state=RANDOM_SEED)

    elif split_type == "random_series":
        train_tresh = int(data.shape[0] * (1.0 - val_perc))
        data_train = data[:train_tresh, ...]
        data_val = data[train_tresh:, ...]

    return data_train, data_val


# normalize data set according to save_cent
def norm_switch(data, save_cent, axes):

    ones_prof = np.ones((1,) + data.shape[1:], dtype=np.float64)
    zero_prof = np.zeros((1,) + data.shape[1:], dtype=np.float64)

    if save_cent == "minmax":
        data_min = np.amin(data, axis=axes, keepdims=True)
        data_max = np.amax(data, axis=axes, keepdims=True)
        norm_sub = data_min * ones_prof
        norm_fac = (data_max - data_min) * ones_prof

    elif save_cent == "l2":
        norm_fac = np.square(data)
        for dim_idx in range(len(axes)):
            norm_fac = np.sum(norm_fac, axis=axes[dim_idx], keepdims=True)
        for dim_idx in range(len(axes)):
            norm_fac[:] /= data.shape[axes[dim_idx]]
        norm_fac = norm_fac * ones_prof
        norm_sub = zero_prof

    else:
        raise ValueError("Invalid choice of save_cent: " + save_cent)

    return norm_sub, norm_fac


# determine how to normalized data given shape, normalize
# assumed to be in NCHW format
def normalize_data_set(data, save_cent, model_dir, network_suffix, norms=None, save_norm=False):

    # calculate norms
    if norms is None:
        num_dims = data.ndim - 2  # ignore samples and channels dimensions
        if num_dims == 1:
            norm_axes = (0, 2)

        elif num_dims == 2:
            norm_axes = (0, 2, 3)

        elif num_dims == 3:
            norm_axes = (0, 2, 3, 4)

        else:
            raise ValueError("Something went wrong with normalizing (data dimensions)")

        norm_sub, norm_fac = norm_switch(data, save_cent, axes=norm_axes)

    # norms are provided
    else:
        norm_sub = norms[0][None, :, :]
        norm_fac = norms[1][None, :, :]

    data = (data - norm_sub) / norm_fac

    if (norms is None) and save_norm:

        norm_sub = np.squeeze(norm_sub, axis=0)
        norm_fac = np.squeeze(norm_fac, axis=0)

        np.save(os.path.join(model_dir, "norm_sub_prof" + network_suffix + ".npy"), norm_sub)
        np.save(os.path.join(model_dir, "norm_fac_prof" + network_suffix + ".npy"), norm_fac)

    return data, norm_sub, norm_fac


def get_shape_tuple(shape_var):

    if type(shape_var) is list:
        if len(shape_var) != 1:
            raise ValueError("Invalid model I/O size")
        else:
            shape_var = shape_var[0]
    elif type(shape_var) is tuple:
        pass
    else:
        raise TypeError("Invalid shape input of type " + str(type(shape_var)))

    return shape_var
