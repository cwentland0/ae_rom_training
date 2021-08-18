import enum
import re
import os
from math import ceil

import numpy as np
from sklearn.model_selection import train_test_split
from hyperopt import tpe, rand
from tensorflow.python.types.core import Value

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

    try:
        # if None passed as default, trust user
        if default_val is None:
            out_val = in_dict[in_key]
        else:
            default_type = type(default_val)
            out_val = default_type(in_dict[in_key])
    except:
        out_val = default_val

    return out_val


def catch_list(in_dict, in_key, default, len_highest=1):

    list_of_lists_flag = type(default[0]) == list

    try:
        inList = in_dict[in_key]

        if len(inList) == 0:
            raise ValueError

        # List of lists
        if list_of_lists_flag:
            val_list = []
            for list_idx in range(len_highest):
                # If default value is None, trust user
                if default[0][0] is None:
                    val_list.append(inList[list_idx])
                else:
                    type_default = type(default[0][0])
                    cast_in_list = [type_default(inVal) for inVal in inList[list_idx]]
                    val_list.append(cast_in_list)

        # Normal list
        else:
            # If default value is None, trust user
            if default[0] is None:
                val_list = inList
            else:
                type_default = type(default[0])
                val_list = [type_default(inVal) for inVal in inList]

    except:
        if list_of_lists_flag:
            val_list = []
            for list_idx in range(len_highest):
                val_list.append(default[0])
        else:
            val_list = default

    return val_list


def read_input_file(input_file):
    """Read text input file for fixed parameters.

    This reads every input parameter from input_file and does some manual checking for FIXED parameters.
    Network parameters are checked when the autoencoders are initialized.
    Variable training parameters are checked right before training.
    """

    # TODO: this is an insanely dumb way of catching required parameters, any better way?

    # raw inputs, no defaults assigned
    input_dict_raw = read_text_file(input_file)

    # this really just checks that all necessary inputs are given and defaults are assigned otherwise
    input_dict = input_dict_raw.copy()

    # path inputs
    input_dict["data_dir"] = input_dict_raw["data_dir"]
    input_dict["data_files_train"] = input_dict_raw["data_files_train"]
    input_dict["data_files_val"] = catch_list(input_dict_raw, "data_files_val", [None])
    input_dict["model_dir"] = os.path.join(input_dict_raw["model_dir"], input_dict_raw["model_label"])
    input_dict["ae_label"] = catch_input(input_dict_raw, "ae_label", "")
    if not os.path.exists(input_dict["model_dir"]):
        os.makedirs(input_dict["model_dir"])

    # data inputs
    input_dict["var_network_idxs"] = list(input_dict_raw["var_network_idxs"])
    input_dict["num_networks"] = len(input_dict["var_network_idxs"])
    input_dict["idx_start_list_train"] = input_dict_raw["idx_start_list_train"]
    input_dict["idx_end_list_train"] = input_dict_raw["idx_end_list_train"]
    input_dict["idx_skip_list_train"] = input_dict_raw["idx_skip_list_train"]
    input_dict["time_init_list_train"] = catch_input(input_dict_raw, "time_init_list_train", [None])
    input_dict["dt_list_train"] = catch_input(input_dict_raw, "dt_list_train", [None])
    input_dict["dt_nondim"] = catch_input(input_dict_raw, "dt_nondim", 1.0e-6)  # time non-dimensionalization
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
    if len(input_dict["time_init_list_train"]) == 1:
        input_dict["time_init_list_train"] *= num_datasets_train
    if len(input_dict["dt_list_train"]) == 1:
        input_dict["dt_list_train"] *= num_datasets_train

    if input_dict["data_files_val"][0] is not None:
        input_dict["idx_start_list_val"] = input_dict_raw["idx_start_list_val"]
        input_dict["idx_end_list_val"] = input_dict_raw["idx_end_list_val"]
        input_dict["idx_skip_list_val"] = input_dict_raw["idx_skip_list_val"]
        input_dict["time_init_list_val"] = catch_input(input_dict_raw, "time_init_list_val", [None])
        input_dict["dt_list_val"] = catch_input(input_dict_raw, "dt_list_val", [None])

        num_datasets_val = len(input_dict["data_files_val"])
        if len(input_dict["idx_start_list_val"]) == 1:
            input_dict["idx_start_list_val"] *= num_datasets_val
        if len(input_dict["idx_end_list_val"]) == 1:
            input_dict["idx_end_list_val"] *= num_datasets_val
        if len(input_dict["idx_skip_list_val"]) == 1:
            input_dict["idx_skip_list_val"] *= num_datasets_val
        if len(input_dict["time_init_list_val"]) == 1:
            input_dict["time_init_list_val"] *= num_datasets_val
        if len(input_dict["dt_list_val"]) == 1:
            input_dict["dt_list_val"] *= num_datasets_val
    
        input_dict["val_perc"] = None

    else:
        input_dict["val_perc"] = input_dict_raw["val_perc"]

    # data preprocessing
    input_dict["split_scheme"] = input_dict_raw["split_scheme"]
    input_dict["centering_scheme"] = input_dict_raw["centering_scheme"]
    input_dict["normal_scheme"] = input_dict_raw["normal_scheme"]

    # global parameters
    input_dict["aerom_type"] = input_dict_raw["aerom_type"]
    input_dict["latent_dim"] = input_dict_raw["latent_dim"]
    if isinstance(input_dict["latent_dim"], int):
        input_dict["latent_dim"] = [input_dict["latent_dim"]]
    if len(input_dict["latent_dim"]) == 1:
        input_dict["latent_dim"] *= input_dict["num_networks"]
    input_dict["train_ae"] = catch_input(input_dict_raw, "train_ae", False)
    input_dict["train_ts"] = catch_input(input_dict_raw, "train_ts", False)
    assert input_dict["train_ae"] or input_dict["train_ts"], "Must set train_ae = True or train_ts = True"

    # misc
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

    # check whether to expect encoded data
    if input_dict["train_ts"] and (not input_dict["train_ae"]):
        data_encoded = True
    else:
        data_encoded = False

    # get training data set
    data_list_raw_train = agg_data_sets(
        input_dict["data_dir"],
        input_dict["data_files_train"],
        input_dict["idx_start_list_train"],
        input_dict["idx_end_list_train"],
        input_dict["idx_skip_list_train"],
        input_dict["data_order"],
        encoded=data_encoded,
        net_idxs=input_dict["var_network_idxs"],
        ae_label=input_dict["ae_label"],
    )

    # get validation data sets, if given
    if input_dict["data_files_val"][0] is not None:
        data_list_raw_val = agg_data_sets(
            input_dict["data_dir"],
            input_dict["data_files_val"],
            input_dict["idx_start_list_val"],
            input_dict["idx_end_list_val"],
            input_dict["idx_skip_list_val"],
            input_dict["data_order"],
            encoded=data_encoded,
            net_idxs=input_dict["var_network_idxs"],
            ae_label=input_dict["ae_label"],
        )
        input_dict["separate_val"] = True
    else:
        data_list_raw_val = None
        input_dict["separate_val"] = False


    # TODO: training/validation split is different for each network, might not be the more rigorous?
    # might be easier to get a fixed index shuffle, apply to all network
    # as this stands right now, no real point to doing centering/normalization separately, since there's
    #   no option to do separate centering/normalization/split schemes for different variables
    data_list_train, data_list_val = [], []
    split_idxs_list_train, split_idxs_list_val = [], []
    for net_idx, var_idxs in enumerate(input_dict["var_network_idxs"]):

        # if data has already been encoded, they were already broken up by network
        if data_encoded:
            data_list_var_train_in = data_list_raw_train[net_idx]
            if data_list_raw_val is not None:
                data_list_var_val_in = data_list_raw_val[net_idx]
            else:
                data_list_var_val_in = None
        else:
            data_list_var_train_in = get_vars_from_data(data_list_raw_train, var_idxs)
            if data_list_raw_val is not None:
                data_list_var_val_in = get_vars_from_data(data_list_raw_val, var_idxs)
            else:
                data_list_var_val_in = None

        # pre-process data
        # includes centering, normalization, and train/validation split
        data_list_var_train, data_list_var_val, split_idxs_list_var_train, split_idxs_list_var_val = preproc_raw_data(
            data_list_var_train_in,
            input_dict["centering_scheme"],
            input_dict["split_scheme"],
            input_dict["normal_scheme"],
            input_dict["model_dir"],
            input_dict["network_suffixes"][net_idx],
            data_list_val_var=data_list_var_val_in,
            val_perc=input_dict["val_perc"],
        )

        # up until now, data has been in NCHW, tranpose if requesting NHWC
        if input_dict["network_order"] == "NHWC":
            num_dims = data_list_var_train[0].ndim - 2  # subtract samples and channels dimensions
            if num_dims == 1:
                trans_axes = (0, 2, 1)
            elif num_dims == 2:
                trans_axes = (0, 2, 3, 1)
            elif num_dims == 3:
                trans_axes = (0, 2, 3, 4, 1)
            for idx, data_arr in enumerate(data_list_var_train):
                data_list_var_train[idx] = np.transpose(data_arr, trans_axes)
            for idx, data_arr in enumerate(data_list_var_val):
                data_list_var_val[idx] = np.transpose(data_arr, trans_axes)
            squeeze_axis = 1
        else:
            squeeze_axis = -1

        # if training on encoded data, squeeze dummy spatial dimension
        if data_encoded:
            for idx, data_arr in enumerate(data_list_var_train):
                data_list_var_train[idx] = np.squeeze(data_arr, axis=squeeze_axis)
            for idx, data_arr in enumerate(data_list_var_val):
                data_list_var_val[idx] = np.squeeze(data_arr, axis=squeeze_axis)

        data_list_train.append(data_list_var_train)
        data_list_val.append(data_list_var_val)
        split_idxs_list_train.append(split_idxs_list_var_train)
        split_idxs_list_val.append(split_idxs_list_var_val)

    return data_list_train, data_list_val, split_idxs_list_train, split_idxs_list_val


def agg_data_sets(
    data_dir,
    data_loc_list,
    idx_start_list,
    idx_end_list,
    idx_skip_list,
    data_order,
    encoded=False,
    net_idxs=None,
    ae_label=None,
):
    """Given list of data locations, aggregate data sets.

    Puts all data in NCHW format.
    """

    if encoded:
        assert (net_idxs is not None) and (
            ae_label is not None
        ), "If aggregating encoded data, must supply net_idxs and ae_label"
        data_dir = os.path.join(data_dir, "encodings", ae_label)

    if encoded:
        data_raw = [[] for _ in range(len(net_idxs))]
    else:
        data_raw = []
    for file_count, data_file in enumerate(data_loc_list):

        data_load = []
        # if encoded data, need data from each network
        if encoded:
            data_name_encoded = data_file[:-4]  # strip .npy
            for var_idxs in net_idxs:
                suffix = ""
                for var_idx in var_idxs:
                    suffix += "_" + str(var_idx)
                data_file_encoded = data_name_encoded + suffix + ".npy"
                data_loc = os.path.join(data_dir, data_file_encoded)
                data_load.append(np.load(data_loc))

        # if full state data, should only be one file
        else:
            data_loc = os.path.join(data_dir, data_file)
            data_load.append(np.load(data_loc))

        num_dims = data_load[0].ndim - 2  # excludes N and C dimensions

        for data_idx, data in enumerate(data_load):

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
                data = np.transpose(data, axes=trans_axes)

            # extract a range of iterations
            if num_dims == 1:
                data = data[idx_start_list[file_count] : idx_end_list[file_count] : idx_skip_list[file_count], :, :]
            elif num_dims == 2:
                data = data[idx_start_list[file_count] : idx_end_list[file_count] : idx_skip_list[file_count], :, :, :]
            elif num_dims == 3:
                data = data[
                    idx_start_list[file_count] : idx_end_list[file_count] : idx_skip_list[file_count], :, :, :, :
                ]

            # aggregate all data sets
            # if encoded, data_raw is a list of lists, sublists correspond to latent variable datasets for each network
            # data_idx could also be net_idx, since this iterates through the data for each network
            if encoded:
                data_raw[data_idx].append(data.copy())
            # if full state data, data_raw is a list, entries are np.ndarrays with state snapshots
            else:
                data_raw.append(data.copy())

    return data_raw


def preproc_raw_data(
    data_list_train_var,
    centering_scheme,
    split_scheme,
    normal_scheme,
    model_dir,
    network_suffix,
    data_list_val_var=None,
    val_perc=None,
):

    data_list_train, data_list_val = [], []
    split_idxs_list_train, split_idxs_list_val = [], []

    
    # if centering by initial condition, center training data here
    if centering_scheme == "init_cond":
        data_list_train_var_cent, _ = center_data_set(
            data_list_train_var, centering_scheme, model_dir, network_suffix, save_cent=True
        )
    else:
        data_list_train_var_cent = data_list_train_var

    # make train/val split from given training data
    if data_list_val_var is None:

        # split, maintain indices
        for dataset_num, data_arr in enumerate(data_list_train_var_cent):
            data_train, data_val, split_idxs_train, split_idxs_val = split_data_set(data_arr, split_scheme, val_perc=val_perc)
            data_list_train.append(data_train)
            data_list_val.append(data_val)
            split_idxs_list_train.append(split_idxs_train)
            split_idxs_list_val.append(split_idxs_val)

    # training/validation split given by files
    else:
        
        # no real splitting of training set, just shuffling
        for dataset_num, data_arr in enumerate(data_list_train_var_cent):
            data_train, _, split_idxs_train, _ = split_data_set(data_arr, split_scheme, val_perc=None)
            data_list_train.append(data_train)
            split_idxs_list_train.append(split_idxs_train)

        # center validation data, but don't need to shuffle
        if centering_scheme == "init_cond":
            data_list_val, _ = center_data_set(
                data_list_val_var, centering_scheme, model_dir, network_suffix, save_cent=True, val=True,
            )
        else:
            data_list_val = data_list_val_var
        split_idxs_list_val = [np.arange(data_arr.shape[0]) for data_arr in data_list_val]

    # if not centering by initial condition, center here
    if centering_scheme != "init_cond":
        data_list_train, cent_prof = center_data_set(
            data_list_train, centering_scheme, model_dir, network_suffix, save_cent=True
        )
        data_list_val, _ = center_data_set(
            data_list_val, centering_scheme, model_dir, network_suffix, cent_prof=cent_prof, save_cent=False
        )

    # normalize training and validation sets separately
    data_list_train, norm_sub_train, norm_fac_train = normalize_data_set(
        data_list_train, normal_scheme, model_dir, network_suffix, save_norm=True
    )
    data_list_val, _, _ = normalize_data_set(
        data_list_val, normal_scheme, model_dir, network_suffix, norms=[norm_sub_train, norm_fac_train], save_norm=False
    )

    return data_list_train, data_list_val, split_idxs_list_train, split_idxs_list_val


def calc_time_values(time_start, dt, idx_start, idx_end, idx_skip, shuffle_idxs=None):

    steps = np.arange(idx_start, idx_end, idx_skip)
    steps -= idx_start
    time_values = time_start + dt * steps
    time_diffs = time_values - time_values[0]

    if shuffle_idxs is not None:
        time_values = time_values[shuffle_idxs]
        time_diffs = time_diffs[shuffle_idxs]

    return time_values, time_diffs


def center_switch(data: list, cent_type):
    """Computes centering profile(s) for data

    data is a list of data arrays

    If cent_type == "init_cond", the output is a list of centering profiles for each data set
    Otherwise, the output is a single array to be applied to all data sets
    """

    # initial condition from each data array
    if cent_type == "init_cond":
        cent_prof = []
        for data_arr in data:
            cent_prof.append(data_arr[[0], ...])

    # time average across all data arrays
    elif cent_type == "mean":
        data_concat = np.concatenate(data, axis=0)
        cent_prof = np.mean(data_concat, axis=0, keepdims=True)

    # no centering
    elif cent_type == "none":
        cent_prof = np.zeros((1,) + data[0].shape[1:], dtype=data[0].dtype)

    else:
        raise ValueError("Invalid choice of cent_type: " + cent_type)

    return cent_prof


def center_data_set(data: list, cent_type, model_dir, network_suffix, cent_prof=None, save_cent=False, val=False):
    """Center data set about some profile.

    data is a list of data arrays assumed to be in NCHW format.
    """

    if cent_prof is None:
        cent_prof = center_switch(data, cent_type)

    for idx, data_arr in enumerate(data):
        if cent_type == "init_cond":
            data[idx] = data_arr - cent_prof[idx]
        else:
            data[idx] = data_arr - cent_prof

    if save_cent:

        if cent_type == "init_cond":
            for idx, prof in enumerate(cent_prof):
                cent_prof_out = np.squeeze(prof, axis=0)
                out_name = "cent_prof_dataset" + str(idx)
                if val:
                    out_name += "_val"
                out_name += network_suffix + ".npy"
                np.save(
                    os.path.join(model_dir, out_name), cent_prof_out
                )
        else:
            if val:
                raise ValueError("Validation set shouldn't be setting out it's own centering profile if not init_cond")
            cent_prof_out = np.squeeze(cent_prof, axis=0)
            np.save(os.path.join(model_dir, "cent_prof" + network_suffix + ".npy"), cent_prof_out)

    return data, cent_prof


def split_data_set(data, split_type, val_perc=None):
    """Split dataset into training and validation sets.

    data is a NumPy array here.
    Also returns indices mapping original dataset snapshot indices to split indices.
    """

    if split_type == "random":
        if val_perc is None:
            idxs_train = np.random.permutation(data.shape[0])
            data_train = data[idxs_train, ...]
            data_val = None
            idxs_val = None
        else:
            indices = np.arange(data.shape[0])
            data_train, data_val, idxs_train, idxs_val = train_test_split(
                data, indices, test_size=val_perc, random_state=RANDOM_SEED
            )

    elif split_type == "series_random":
        if val_perc is None:
            raise ValueError("series_random got val_perc = None")
        train_tresh = int(data.shape[0] * (1.0 - val_perc))
        data_train = data[:train_tresh, ...]
        data_val = data[train_tresh:, ...]
        idxs_train = np.arange(train_tresh)
        idxs_val = np.arange(train_tresh, data.shape[0])

    else:
        raise ValueError("Invalid split_scheme: " + str(split_type))

    return data_train, data_val, idxs_train, idxs_val


def hankelize(data: list, seq_length, seq_step=1):
    """Arrange data snapshots into windows

    data is assumed to be a list of NumPy arrays.
    Returns a list of Hankelized matrices.
    """

    data_seqs = []
    num_dims = data[0].ndim

    for data_arr in data:
        num_snaps = data_arr.shape[0]

        # accommodate single-value sequences, actually need to keep this for layer inputs
        if num_dims == 1:
            data_arr = np.expand_dims(data_arr, axis=-1)

        # extract windows
        num_seqs = ceil((num_snaps - seq_length + 1) / seq_step)
        data_seqs.append(np.zeros((num_seqs, seq_length,) + data_arr.shape[1:], dtype=data_arr.dtype))
        for seq_idx in range(num_seqs):
            idx_start = seq_idx * seq_step
            idx_end = idx_start + seq_length
            data_seqs[-1][seq_idx, ...] = data_arr[idx_start:idx_end, ...]

        # account for final window
        if idx_end != num_snaps:
            data_seqs[-1][-1, ...] = data_arr[-seq_length:, ...]

    return data_seqs


def window(data: list, seq_length, pred_length=1, seq_step=1):
    """Similar to Hankelization, but returns prediction ``label'' data.
    
    data is assumed to be a list of NumPy arrays.
    Gets ``labels'' from the next pred_length snapshots after seq_length
    """

    # TODO: just roll this into hankelize()

    data_seqs = []
    pred_seqs = []
    num_dims = data[0].ndim

    for data_arr in data:
        num_snaps = data_arr.shape[0]

        # accommodate single-value sequences, actually need to keep this for layer inputs
        if num_dims == 1:
            data_arr = np.expand_dims(data_arr, axis=-1)

        num_seqs = ceil((num_snaps - seq_length - pred_length + 1) / seq_step)
        data_seqs.append(np.zeros((num_seqs, seq_length,) + data_arr.shape[1:], dtype=data_arr.dtype))
        pred_seqs.append(np.zeros((num_seqs, pred_length,) + data_arr.shape[1:], dtype=data_arr.dtype))
        for seq_idx in range(num_seqs):
            idx_start = seq_idx * seq_step
            idx_end = idx_start + seq_length
            data_seqs[-1][seq_idx, ...] = data_arr[idx_start:idx_end, ...]
            pred_seqs[-1][seq_idx, ...] = data_arr[idx_end : idx_end + pred_length, ...]

        # account for final window
        if idx_end != num_snaps - pred_length:
            # TODO: if you want to roll this into hankelize, slice to None instead of -pred_length
            data_seqs[-1][-1, ...] = data_arr[-seq_length - pred_length : -pred_length, ...]
            pred_seqs[-1][-1, ...] = data_arr[-pred_length:, ...]

    return data_seqs, pred_seqs


def sequencize(
    data_list_train,
    split_idxs_train,
    data_list_val,
    split_idxs_val,
    seq_lookback,
    seq_step,
    pred_length=1,
    hankelize_data=False,
    separate_val=False,
):

    # need to unshuffle, make sequences, then separate sequences out again
    # put consecutive data back together again
    data_list_full = []
    for idx, data_train in enumerate(data_list_train):
        
        split_idxs_train_arr = split_idxs_train[idx]
        
        # validation set should NOT have been shuffled, don't need to deal with it
        if separate_val:
            data_full = np.zeros(data_train.shape, dtype=data_train.dtype)
            data_full[split_idxs_train_arr, ...] = data_train.copy()
            data_list_full.append(data_full.copy())

        # input data was split, shuffled
        else:
            split_idxs_val_arr = split_idxs_val[idx]
            data_val = data_list_val[idx]
            num_snaps = data_train.shape[0] + data_val.shape[0]
            data_full = np.zeros((num_snaps,) + data_train.shape[1:], dtype=data_train.dtype)
            data_full[split_idxs_train_arr, ...] = data_train.copy()
            data_full[split_idxs_val_arr, ...] = data_val.copy()
            data_list_full.append(data_full.copy())

    # window data, making inputs and labels
    if hankelize_data:
        data_seqs_in = hankelize(data_list_full, seq_lookback + pred_length, seq_step)
        if separate_val:
            data_seqs_in_val = hankelize(data_list_val, seq_lookback + pred_length, seq_step)
    else:
        data_seqs_in, data_seqs_out = window(data_list_full, seq_lookback, pred_length=pred_length, seq_step=seq_step)
        if separate_val:
            data_seqs_in_val, data_seqs_out_val = window(data_list_val, seq_lookback, pred_length=pred_length, seq_step=seq_step)

    # redistribute training and validation data
    data_list_train_seqs = []
    data_list_val_seqs = []
    data_list_train_seqs_pred = []
    data_list_val_seqs_pred = []
    for idx, data_seqs in enumerate(data_seqs_in):

        # need to exclude 0:seq_lookback, subtract seq_lookback to get new sorting indices
        # NOTE: assume_unique maintains shuffle
        idxs_train = np.setdiff1d(split_idxs_train[idx], np.arange(0, seq_lookback), assume_unique=True) - seq_lookback
        data_list_train_seqs.append(data_seqs[idxs_train, ...])

        if not separate_val:
            idxs_val = np.setdiff1d(split_idxs_val[idx], np.arange(0, seq_lookback), assume_unique=True) - seq_lookback
            data_list_val_seqs.append(data_seqs[idxs_val, ...])

        # make separate prediction matrix if windowing data
        if not hankelize_data:
            data_seqs_pred = data_seqs_out[idx]
            data_list_train_seqs_pred.append(data_seqs_pred[idxs_train, ...])
            if not separate_val:
                data_list_val_seqs_pred.append(data_seqs_pred[idxs_val, ...])

    # again, validation set should never have been sorted in the first place if using separate validation sets
    if separate_val:
        data_list_val_seqs = data_seqs_in_val
        if not hankelize_data:
            data_list_val_seqs_pred = data_seqs_out_val

    # concatenate data
    data_train_input = np.concatenate(data_list_train_seqs, axis=0)
    data_val_input = np.concatenate(data_list_val_seqs, axis=0)

    if not hankelize_data:
        data_train_output = np.concatenate(data_list_train_seqs_pred, axis=0)
        data_val_output = np.concatenate(data_list_val_seqs_pred, axis=0)
    else:
        data_train_output = None
        data_val_output = None

    return data_train_input, data_train_output, data_val_input, data_val_output


def norm_switch(data: list, norm_type, axes):
    """Compute normalization profile

    Here, data is a single array concatenated along the time axis.
    """

    ones_prof = np.ones((1,) + data.shape[1:], dtype=np.float64)
    zero_prof = np.zeros((1,) + data.shape[1:], dtype=np.float64)

    if norm_type == "minmax":
        data_min = np.amin(data, axis=axes, keepdims=True)
        data_max = np.amax(data, axis=axes, keepdims=True)
        norm_sub = data_min * ones_prof
        norm_fac = (data_max - data_min) * ones_prof

    elif norm_type == "l2":
        norm_fac = np.square(data)
        for dim_idx in range(len(axes)):
            norm_fac = np.sum(norm_fac, axis=axes[dim_idx], keepdims=True)
        for dim_idx in range(len(axes)):
            norm_fac[:] /= data.shape[axes[dim_idx]]
        norm_fac = norm_fac * ones_prof
        norm_sub = zero_prof

    elif norm_type == "none":
        norm_fac = ones_prof
        norm_sub = zero_prof

    else:
        raise ValueError("Invalid choice of norm_type: " + norm_type)

    return norm_sub, norm_fac


# determine how to normalized data given shape, normalize
# assumed to be in NCHW format
def normalize_data_set(data: list, norm_type, model_dir, network_suffix, norms=None, save_norm=False):
    """Normalize data

    data is a list of data arrays assumed to be in NCHW format
    """

    # calculate norms
    if norms is None:
        num_dims = data[0].ndim - 2  # ignore samples and channels dimensions
        if num_dims == 1:
            norm_axes = (0, 2)

        elif num_dims == 2:
            norm_axes = (0, 2, 3)

        elif num_dims == 3:
            norm_axes = (0, 2, 3, 4)

        else:
            raise ValueError("Something went wrong with normalizing (data dimensions)")

        data_concat = np.concatenate(data, axis=0)
        norm_sub, norm_fac = norm_switch(data_concat, norm_type, axes=norm_axes)

    # norms are provided
    else:
        norm_sub = norms[0]
        norm_fac = norms[1]

    for idx, data_arr in enumerate(data):
        data[idx] = (data_arr - norm_sub) / norm_fac

    if (norms is None) and save_norm:

        norm_sub_out = np.squeeze(norm_sub, axis=0)
        norm_fac_out = np.squeeze(norm_fac, axis=0)

        np.save(os.path.join(model_dir, "norm_sub_prof" + network_suffix + ".npy"), norm_sub_out)
        np.save(os.path.join(model_dir, "norm_fac_prof" + network_suffix + ".npy"), norm_fac_out)

    return data, norm_sub, norm_fac


def remove_prefix(text: str, prefix: str):
    if text.startswith(prefix):
        return text[len(prefix) :]
    return text


def get_shape_tuple(shape_var):

    if type(shape_var) is list:
        if len(shape_var) == 1:
            return shape_var[0]
        else:
            return shape_var

    elif type(shape_var) is tuple:
        return shape_var
    else:
        raise TypeError("Invalid shape input of type " + str(type(shape_var)))


def seed_rng(mllib):
    """Seeds NumPy random number generator and ML library random number generator.
    
    More or less resets RNG to a ground state, useful for ensuring model training is the same run to run.
    """

    np.random.seed(RANDOM_SEED)  # seed NumPy RNG
    mllib.seed_rng()
