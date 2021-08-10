import os

import numpy as np

import ae_rom_training.ml_library as ml

# ----- start user input

mllib_name = "tfkeras"

model_dir = (
    "/scratch/kdur_root/kdur/chriswen/1d_transient_flame_1024cells/forced/models/new_results/"
)
model_label = "amp_0p01_freq_100k/baseline_ae/centNone_normMinMax/k20/primVars_samp10_vector_k20_actSwish_filt16-32-64_stride2_kern8_batch10"

data_dir = (
    "/scratch/kdur_root/kdur/chriswen/1d_transient_flame_1024cells/forced/FOM/amp_0p01_freq_100k/freq_100000/unsteady_field_results"
)
data_files = ["sol_prim_FOM.npy"]

var_network_idxs = [[0, 1, 2, 3]]

idx_start_list = [0]
idx_end_list = [None]
idx_skip_list = [1]

data_order = "CHWN"
network_order = "NHWC"

# ----- end user input

out_base = os.path.join(data_dir, "encodings", model_label)
if not os.path.isdir(out_base):
    os.makedirs(out_base)

model_dir = os.path.join(model_dir, model_label)

# get ML library
mllib = ml.get_ml_library(mllib_name, False)

num_datasets = len(data_files)
num_networks = len(var_network_idxs)

# get network input file suffixes
suffix_list = []
for var_idxs in var_network_idxs:
    suffix = ""
    for var_idx in var_idxs:
        suffix += "_" + str(var_idx)
    suffix_list.append(suffix)

# get centering and normalization data
norm_fac_prof_list = []
norm_sub_prof_list = []
cent_prof_list = [[]] * num_networks
for net_idx in range(num_networks):

    norm_fac_file = os.path.join(model_dir, "norm_fac_prof" + suffix_list[net_idx] + ".npy")
    norm_fac_prof = np.load(norm_fac_file)
    norm_fac_prof_list.append(norm_fac_prof.copy())

    norm_sub_file = os.path.join(model_dir, "norm_sub_prof" + suffix_list[net_idx] + ".npy")
    norm_sub_prof = np.load(norm_sub_file)
    norm_sub_prof_list.append(norm_sub_prof.copy())

    not_ic = False
    for file_idx in range(num_datasets):
        cent_file = os.path.join(model_dir, "cent_prof_dataset" + str(file_idx) + suffix_list[net_idx] + ".npy")
        try:
            cent_prof = np.load(cent_file)
        except FileNotFoundError as e:
            if file_idx == 0:
                not_ic = True
                break
            else:
                print(e)
                raise FileNotFoundError
        cent_prof_list[net_idx].append(cent_prof.copy())
    if not_ic:
        cent_file = os.path.join(model_dir, "cent_prof" + suffix_list[net_idx] + ".npy")
        try:
            cent_prof = np.load(cent_file)
        except FileNotFoundError as e:
            print("Could not find IC centering file OR normal centering file")
            print(e)
            raise FileNotFoundError
        cent_prof_list[net_idx].append(cent_prof.copy())

# get data and format
data_list = [[]] * num_networks
for file_idx, data_file in enumerate(data_files):
    print("Loading data set " + str(file_idx + 1) + "/" + str(num_datasets))

    file_name = os.path.join(data_dir, data_file)
    data = np.load(file_name)
    num_dims = data.ndim - 2
    # For now, everything goes to NCHW, will get transposed to network_order
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
    data = data[idx_start_list[file_idx] : idx_end_list[file_idx] : idx_skip_list[file_idx], ...]

    # separate data by network, standardize
    # NOTE: standardization profiles are always in CHW format
    for net_idx, var_idxs in enumerate(var_network_idxs):
        # get standardization data
        data_var_norm = (
            data[:, var_idxs, ...]
            - cent_prof_list[net_idx][file_idx][None, ...]
            - norm_sub_prof_list[net_idx][None, ...]
        ) / norm_fac_prof_list[net_idx][None, ...]
        data_list[net_idx].append(data_var_norm.copy())

# transpose datasets to network_order
if network_order == "NHWC":
    if num_dims == 1:
        trans_axes = (0, 2, 1)
    elif num_dims == 2:
        trans_axes = (0, 2, 3, 1)
    elif num_dims == 3:
        trans_axes = (0, 2, 3, 4, 1)
for net_idx in range(num_networks):
    for file_idx in range(num_datasets):
        data_list[net_idx][file_idx] = np.transpose(data_list[net_idx][file_idx], trans_axes)

# loop over models
latent_vars_list = [[]] * num_networks
for net_idx in range(num_networks):

    print("Encoding network " + str(net_idx + 1) + "/" + str(num_networks))
    # get encoder
    encoder_file = "encoder"
    for var_idx in var_network_idxs[net_idx]:
        encoder_file += "_" + str(var_idx)
    encoder = mllib.load_model(model_dir, encoder_file)

    # encode data
    for file_idx in range(num_datasets):
        latent_vars = (mllib.eval_model(encoder, data_list[net_idx][file_idx])).astype(np.float64)
        latent_vars_list[net_idx].append(latent_vars.copy())

# put latent variable data back together
# NOTE: up to user to properly split data by network in ae_rom_training
for file_idx in range(num_datasets):

    print("Saving data file " + str(file_idx + 1) + "/" + str(num_datasets))

    # NOTE: latent dimension is treated as channels dimension, adding spurious spatial dimension
    for net_idx in range(num_networks):
        latent_vars = latent_vars_list[net_idx][file_idx]
        if network_order == "NHWC":
            latent_vars = latent_vars[:, None, :]
            concat_axis = 1
        elif network_order == "NCHW":
            latent_vars = latent_vars[:, :, None]
            concat_axis = 2

        # make subdirectories
        data_file_dir, data_file_name = os.path.split(data_files[file_idx])
        if data_file_dir != "":
            out_dir = os.path.join(out_base, data_file_dir)
            if not os.path.isdir:
                os.makedirs(out_dir)
        else:
            out_dir = out_base

        # write to disk
        file_prefix = data_file_name[:-4]  # strip *.npy
        suffix = ""
        for var_idx in var_network_idxs[net_idx]:
            suffix += "_" + str(var_idx)
        file_name = file_prefix + "_encoded" + suffix + ".npy"
        file_name = os.path.join(out_dir, file_name)
        np.save(file_name, latent_vars)

print("Finished!")
