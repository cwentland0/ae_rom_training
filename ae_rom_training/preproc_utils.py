import os

import numpy as np
from tensorflow.keras.regularizers import l1, l2
from sklearn.model_selection import train_test_split


# given list of data locations, aggregate data sets
# puts all data in NCHW format
def agg_data_sets(data_dir, data_loc_list, idx_start_list, idx_end_list, idx_skip_list, data_order):
    data_raw = []
    for file_count, data_file in enumerate(data_loc_list):
        data_loc = os.path.join(data_dir, data_file)
        data_load = np.load(data_loc)

        num_dims = data_load.ndim - 2  # excludes N and C dimensions

        # NOTE: Keras models only accept NCHW or NHWC.
        # 	Additionally, when running on CPUs, convolutional layers only accept NHWC
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


def preproc_param_objs(space, num_dims, feat_shape, data_format):
    # weight regularization switch

    if space["kernel_reg_type"] is not None:
        space["kernel_reg"] = regularization_switch(space["kernel_reg_type"], space["kernel_reg_val"])
    else:
        space["kernel_reg"] = None

    # activation regularization switch
    if space["act_reg_type"] is not None:
        space["act_reg"] = regularization_switch(space["act_reg_type"], space["act_reg_val"])
    else:
        space["act_reg"] = None

    # bias regularization switch
    if space["bias_reg_type"] is not None:
        space["bias_reg"] = regularization_switch(space["bias_reg_type"], space["bias_reg_val"])
    else:
        space["bias_reg"] = None

    # strides and kernel sizes need to be tuples
    space["stride_list_tuple"] = space["stride_list"]
    space["kern_size_fixed_tuple"] = (int(space["kern_size_fixed"]),) * num_dims

    # check whether all-convolutional network can be built from given inputs
    if space["all_conv"]:

        if data_format == "channels_last":
            spatial_dims = list(reversed(feat_shape[:-1]))
        else:
            spatial_dims = list(reversed(feat_shape[1:]))

        num_conv_layers = space["num_conv_layers"]
        x_strides = [x[0] for x in space["stride_list_tuple"]]
        x_dim_final = spatial_dims[0] / np.prod(x_strides[:num_conv_layers])
        dim_final = x_dim_final
        if num_dims == 2:
            y_strides = [x[1] for x in space["stride_list_tuple"]]
            y_dim_final = spatial_dims[1] / np.prod(y_strides[:num_conv_layers])
            dim_final *= y_dim_final
        if num_dims == 3:
            z_strides = [x[2] for x in space["stride_list_tuple"]]
            z_dim_final = spatial_dims[2] / np.prod(z_strides[:num_conv_layers])
            dim_final *= z_dim_final

        filt_final = space["latent_dim"] / dim_final
        assert filt_final.is_integer(), "Cannot make final layer all-convolutional"
        space["filt_final"] = int(filt_final)

    return space


# return regularizer objects
def regularization_switch(reg_type, reg_mult):

    if reg_type == "l2":
        return l2(reg_mult)
    elif reg_type == "l1":
        return l1(reg_mult)
    else:
        raise ValueError("Invalid regularization type:" + str(reg_type))


# TODO: actually implement switches properly
# TODO: handle multiple datasets correctly
def preproc_raw_data(
    data_list_train, data_list_val, centering_scheme, normal_scheme, val_perc, model_dir, network_suffix
):

    # make train/val split from given training data
    if data_list_val is None:

        # concatenate samples after centering
        for dataset_num, data_arr in enumerate(data_list_train):
            data_in = center_data_set(data_arr, centering_scheme, model_dir, network_suffix, save_cent=True)
            data_in_train, data_in_val = train_test_split(data_in, test_size=val_perc, random_state=24)
            if dataset_num == 0:
                data_train = data_in_train.copy()
                data_val = data_in_val.copy()
            else:
                data_train = np.append(data_train, data_in_train, axis=0)
                data_val = np.append(data_val, data_in_val, axis=0)

    else:
        # aggregate training samples after centering
        for dataset_num, data_arr in enumerate(data_list_train):
            data_in_train = center_data_set(data_arr, centering_scheme, model_dir, network_suffix, save_cent=True)
            if dataset_num == 0:
                data_train = data_in_train.copy()
            else:
                data_train = np.append(data_train, data_in_train, axis=0)
        # shuffle training data to avoid temporal/dataset bias
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
        np.save(os.path.join(model_dir, "cent_prof_temp" + network_suffix + ".npy"), cent_prof)

    return data


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

        np.save(os.path.join(model_dir, "norm_sub_prof_temp" + network_suffix + ".npy"), norm_sub)
        np.save(os.path.join(model_dir, "norm_fac_prof_temp" + network_suffix + ".npy"), norm_fac)

    return data, norm_sub, norm_fac
