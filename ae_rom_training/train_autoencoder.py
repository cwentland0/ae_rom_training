import os

import numpy as np
from tensorflow.keras.models import Model
import tensorflow.keras.backend as K
from hyperopt import STATUS_OK
import time
import pickle

from ae_rom_training.preproc_utils import preproc_param_objs, preproc_raw_data
from ae_rom_training.cnn_utils import get_loss

FITVERBOSITY = 2  # 1 for progress bar, 2 for no progress bar



# construct convolutional autoencoder
def build_model(space, mllib, feat_shape, num_dims, data_format, explicit_batch):

    # ----- NETWORK DEFINITION -----

    # alter format of some inputs
    space = preproc_param_objs(space, num_dims, feat_shape, data_format)

    K.set_floatx("float" + str(int(space["layer_precision"])))  # set network numerical precision

    # implicit batch for tf.keras training
    if explicit_batch == 0:
        batch_size = None
    # batch size one for single inference
    elif explicit_batch == 1:
        batch_size = 1
    # explicit batch size networks for Jacobian inference
    else:
        batch_size = explicit_batch  # just for CAE compatibility for decoder Jacobian

    if data_format == "channels_first":
        num_channels = feat_shape[0]
    else:
        num_channels = feat_shape[-1]

    # handle some issues with HyperOpt making floats instead of ints
    num_conv_layers = int(space["num_conv_layers"])
    num_filt_start = int(space["num_filt_start"])
    filt_growth_mult = int(space["filt_growth_mult"])
    kern_size_fixed = space["kern_size_fixed_tuple"]  # this is already handled in preprocUtils

    input_encoder = mllib.get_input_layer(feat_shape, batch_size=batch_size)

    if space["all_conv"]:
        num_conv_layers += 1  # extra layer for all-convolutional network

    # construct encoder portion of autoencoder
    def build_encoder():

        x = input_encoder

        # define sequential convolution layers
        num_filters = num_filt_start
        for conv_num in range(0, num_conv_layers):

            num_kernels = kern_size_fixed

            if space["all_conv"] and (conv_num == (num_conv_layers - 1)):
                num_strides = 1
            else:
                num_strides = space["stride_list_tuple"][conv_num]

            x = mllib.get_conv_layer(
                x,
                num_dims,
                num_filters,
                num_kernels,
                num_strides,
                data_format,
                activation=space["activation_func"],
                padding="same",
                kern_reg=space["kernel_reg"],
                act_reg=space["act_reg"],
                bias_reg=space["bias_reg"],
                kern_init=space["kernel_init_dist"],
                bias_init=space["bias_init_dist"],
            )

            if space["all_conv"] and (conv_num == (num_conv_layers - 2)):
                num_filters = space["filt_final"]
            else:
                num_filters = num_filters * filt_growth_mult

        # flatten before dense layer
        shape_before_flatten = x.shape.as_list()[1:]
        x = mllib.get_flatten_layer(x)

        # without dense layer
        if not space["all_conv"]:

            # set dense layer, if specified
            x = mllib.get_dense_layer(
                x,
                int(space["latent_dim"]),
                activation=space["activation_func"],
                kern_reg=space["kernel_reg"],
                act_reg=space["act_reg"],
                bias_reg=space["bias_reg"],
                kern_init=space["kernel_init_dist"],
                bias_init=space["bias_init_dist"],
            )

        # NOTE: this reshape is for conformity with TensorRT
        # TODO: add a flag if making for TensorRT, otherwise this is pointless
        if num_dims == 2:
            x = mllib.get_reshape_layer(x, (1, 1, int(space["latent_dim"])))
        if num_dims == 3:
            x = mllib.get_reshape_layer(x, (1, 1, 1, int(space["latent_dim"])))

        return Model(input_encoder, x), shape_before_flatten

    # get info necessary to define first few layers of decoder
    encoder, shape_before_flatten = build_encoder()
    dim_before_flatten = np.prod(shape_before_flatten)

    # construct decoder portion of autoencoder
    def build_decoder():

        # hard to a priori know the final convolutional layer output shape, so just copy from encoder output shape
        decoder_input_shape = encoder.layers[-1].output_shape[1:]

        input_decoder = mllib.get_input_layer(decoder_input_shape, batch_size=batch_size)

        x = input_decoder

        if not space["all_conv"]:
            # dense layer
            x = mllib.get_dense_layer(
                x,
                dim_before_flatten,
                activation=space["activation_func"],
                kern_reg=space["kernel_reg"],
                act_reg=space["act_reg"],
                bias_reg=space["bias_reg"],
                kern_init=space["kernel_init_dist"],
                bias_init=space["bias_init_dist"],
            )

        # reverse flattening for input to convolutional layer
        x = mllib.get_reshape_layer(x, shape_before_flatten)

        # define sequential transpose convolutional layers
        num_filters = num_filt_start * filt_growth_mult ** (num_conv_layers - 2)
        for deconv_num in range(0, num_conv_layers):

            # make sure last transpose convolution has a linear activation and as many filters as original channels
            if deconv_num == (num_conv_layers - 1):
                deconv_act = space["final_activation_func"]
                num_filters = num_channels
            else:
                deconv_act = space["activation_func"]

            num_kernels = kern_size_fixed
            if space["all_conv"]:
                if deconv_num == 0:
                    num_strides = 1
                else:
                    num_strides = space["stride_list_tuple"][deconv_num - 1]
            else:
                num_strides = space["stride_list_tuple"][deconv_num]

            x = mllib.get_trans_conv_layer(
                x,
                num_dims,
                num_filters,
                num_kernels,
                num_strides,
                data_format,
                activation=deconv_act,
                padding="same",
                kern_reg=space["kernel_reg"],
                act_reg=space["act_reg"],
                bias_reg=space["bias_reg"],
                kern_init=space["kernel_init_dist"],
                bias_init=space["bias_init_dist"],
            )

            num_filters = int(num_filters / filt_growth_mult)

        return Model(input_decoder, x)

    decoder = build_decoder()
    return Model(input_encoder, decoder(encoder(input_encoder)))


# train the model
# accepts untrained model and optimization params
# returns trained model and loss metrics
def train_cae(space, model, data_train, data_val):

    encoder = model.layers[-2]
    encoder.summary()
    decoder = model.layers[-1]
    decoder.summary()

    # breakpoint()

    loss = get_loss(space["loss_func"])
    model.compile(optimizer=Adam(learning_rate=float(space["learn_rate"])), loss=loss)

    # define callbacks
    callback_list = []
    early_stop = EarlyStopping(patience=int(space["es_patience"]), restore_best_weights=True)
    callback_list.append(early_stop)

    # train model
    model.fit(
        x=data_train,
        y=data_train,
        batch_size=int(space["batch_size"]),
        epochs=int(space["max_epochs"]),
        validation_data=(data_val, data_val),
        verbose=FITVERBOSITY,
        callbacks=callback_list,
    )

    loss_train = model.evaluate(x=data_train, y=data_train, verbose=0)
    loss_val = model.evaluate(x=data_val, y=data_val, verbose=0)

    # repeat this just in case encoder/decoder memory reference has changed
    encoder = model.layers[-2]
    decoder = model.layers[-1]

    return encoder, decoder, loss_train, loss_val


# check if current trained model is best model produced so far
# if best, save to disk and update best validation loss
def check_cae(encoder, decoder, loss_val, space, model_dir, network_suffix):

    loss_loc = os.path.join(model_dir, "valLoss" + network_suffix + ".dat")

    # open minimum loss file
    try:
        with open(loss_loc) as f:
            min_loss = float(f.read().strip())
    except FileNotFoundError:
        print("First iteration, writing best model")
        min_loss = loss_val + 1.0  # first time, guarantee that loss is written to file

    # if current model validation loss beats previous best, overwrite
    if loss_val < min_loss:
        print("New best found! Overwriting...")

        encoder.save(os.path.join(model_dir, "encoder" + network_suffix + ".h5"))
        decoder.save(os.path.join(model_dir, "decoder" + network_suffix + ".h5"))

        space_loc = os.path.join(model_dir, "paramSpace" + network_suffix + ".pickle")
        with open(space_loc, "wb") as f:
            pickle.dump(space, f)

        with open(loss_loc, "w") as f:
            f.write(str(loss_val))

        os.rename(
            os.path.join(model_dir, "cent_prof_temp" + network_suffix + ".npy"),
            os.path.join(model_dir, "cent_prof" + network_suffix + ".npy"),
        )
        os.rename(
            os.path.join(model_dir, "norm_sub_prof_temp" + network_suffix + ".npy"),
            os.path.join(model_dir, "norm_sub_prof" + network_suffix + ".npy"),
        )
        os.rename(
            os.path.join(model_dir, "norm_fac_prof_temp" + network_suffix + ".npy"),
            os.path.join(model_dir, "norm_fac_prof" + network_suffix + ".npy"),
        )

    else:
        os.remove(os.path.join(model_dir, "cent_prof_temp" + network_suffix + ".npy"))
        os.remove(os.path.join(model_dir, "norm_sub_prof_temp" + network_suffix + ".npy"))
        os.remove(os.path.join(model_dir, "norm_fac_prof_temp" + network_suffix + ".npy"))
