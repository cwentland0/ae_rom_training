import os

import numpy as np
from tensorflow.keras.layers import Dense, Input, Flatten, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from hyperopt import STATUS_OK
import time
import pickle

from preproc_utils import preproc_param_objs, preproc_raw_data
from cnn_utils import set_conv_layer, get_loss

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
FITVERBOSITY = 2  # 1 for progress bar, 2 for no progress bar


# "driver" objective function
# preprocesses data set, builds the network, trains the network, and returns training metrics
def objective_func(space, data_list_train, data_list_val, data_format, model_dir, network_suffix):

    t_start = time.time()

    # pre-process data
    # includes centering, normalization, and train/validation split
    data_train, data_val = preproc_raw_data(
        data_list_train,
        data_list_val,
        space["centering_scheme"],
        space["normal_scheme"],
        space["val_perc"],
        model_dir,
        network_suffix,
    )

    num_dims = data_train.ndim - 2  # assumed spatially-oriented data, so subtract samples and channels dimensions

    # up until now, data has been in NCHW, tranpose if requesting NHWC
    if data_format == "channels_last":
        if num_dims == 1:
            trans_axes = (0, 2, 1)
        elif num_dims == 2:
            trans_axes = (0, 2, 3, 1)
        elif num_dims == 3:
            trans_axes = (0, 2, 3, 4, 1)
        data_train = np.transpose(data_train, trans_axes)
        data_val = np.transpose(data_val, trans_axes)

    # build network
    feat_shape = data_train.shape[1:]  # shape of each sample (including channels)
    model = build_model(space, feat_shape, num_dims, data_format, 0)  # must be implicit batch for training

    # train network
    # returns trained model and loss metrics
    encoder, decoder, loss_train, loss_val = train_cae(space, model, data_train, data_val)

    # check if best validation loss
    # save to disk if best, update best validation loss so far
    check_cae(encoder, decoder, loss_val, space, model_dir, network_suffix)

    # return optimization info dictionary
    return {
        "loss": loss_train,  # training loss at end of training
        "true_loss": loss_val,  # validation loss at end of training
        "status": STATUS_OK,  # check for correct exit
        "eval_time": time.time() - t_start,  # time (in seconds) to train model
    }


# construct convolutional autoencoder
def build_model(space, feat_shape, num_dims, data_format, explicit_batch):

    # ----- NETWORK DEFINITION -----

    # alter format of some inputs
    space = preproc_param_objs(space, num_dims, feat_shape, data_format)

    K.set_floatx("float" + str(int(space["layer_precision"])))  # set network numerical precision

    feat_shape_list = list(feat_shape)
    # implicit batch for tf.keras training
    if explicit_batch == 0:
        feat_shape_list.insert(0, None)
    # batch size one for single inference
    elif explicit_batch == 1:
        feat_shape_list.insert(0, 1)
    # explicit batch size networks for Jacobian inference
    else:
        feat_shape_list.insert(0, explicit_batch)  # just for CAE compatibility for decoder Jacobian

    feat_shape = tuple(feat_shape_list)
    if data_format == "channels_first":
        num_channels = feat_shape[1]
    else:
        num_channels = feat_shape[-1]

    # handle some issues with HyperOpt making floats instead of ints
    num_conv_layers = int(space["num_conv_layers"])
    num_filt_start = int(space["num_filt_start"])
    filt_growth_mult = int(space["filt_growth_mult"])
    kern_size_fixed = space["kern_size_fixed_tuple"]  # this is already handled in preprocUtils

    input_encoder = Input(batch_shape=feat_shape, name="inputEncode")

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

            x = set_conv_layer(
                inputVals=x,
                conv_num=conv_num,
                dims=num_dims,
                num_filters=num_filters,
                num_kernels=num_kernels,
                num_strides=num_strides,
                data_format=data_format,
                padding="same",
                kern_reg=space["kernel_reg"],
                act_reg=space["act_reg"],
                bias_reg=space["bias_reg"],
                activation=space["activation_func"],
                kernel_init=space["kernel_init_dist"],
                bias_init=space["bias_init_dist"],
                trans=False,
            )

            if space["all_conv"] and (conv_num == (num_conv_layers - 2)):
                num_filters = space["filt_final"]
            else:
                num_filters = num_filters * filt_growth_mult

        # flatten before dense layer
        shape_before_flatten = x.shape.as_list()[1:]
        x = Flatten(name="flatten")(x)

        # without dense layer
        if not space["all_conv"]:

            # set dense layer, if specified
            x = Dense(
                int(space["latent_dim"]),
                activation=space["activation_func"],
                kernel_regularizer=space["kernel_reg"],
                activity_regularizer=space["act_reg"],
                bias_regularizer=space["bias_reg"],
                kernel_initializer=space["kernel_init_dist"],
                bias_initializer=space["bias_init_dist"],
                name="fcnConv",
            )(x)

        # NOTE: this reshape is for conformity with TensorRT
        # TODO: add a flag if making for TensorRT, otherwise this is pointless
        if num_dims == 2:
            x = Reshape((1, 1, int(space["latent_dim"])))(x)
        if num_dims == 3:
            x = Reshape((1, 1, 1, int(space["latent_dim"])))(x)

        return Model(input_encoder, x), shape_before_flatten

    # get info necessary to define first few layers of decoder
    encoder, shape_before_flatten = build_encoder()
    dim_before_flatten = np.prod(shape_before_flatten)

    # construct decoder portion of autoencoder
    def build_decoder():

        # hard to a priori know the final convolutional layer output shape, so just copy from encoder output shape
        decoder_input_shape = encoder.layers[-1].output_shape[1:]
        decode_input_shape_list = list(decoder_input_shape)

        # implicit batch size for tf.keras training
        if explicit_batch == 0:
            decode_input_shape_list.insert(0, None)
        else:
            decode_input_shape_list.insert(0, explicit_batch)  # for explicit-size decoder batch Jacobian

        decoder_input_shape = tuple(decode_input_shape_list)
        input_decoder = Input(batch_shape=decoder_input_shape, name="inputDecode")

        x = input_decoder

        if not space["all_conv"]:
            # dense layer
            x = Dense(
                dim_before_flatten,
                activation=space["activation_func"],
                kernel_regularizer=space["kernel_reg"],
                activity_regularizer=space["act_reg"],
                bias_regularizer=space["bias_reg"],
                kernel_initializer=space["kernel_init_dist"],
                bias_initializer=space["bias_init_dist"],
                name="fcnDeconv",
            )(x)

        # reverse flattening for input to convolutional layer
        x = Reshape(target_shape=shape_before_flatten, name="reshapeConv")(x)

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

            x = set_conv_layer(
                inputVals=x,
                conv_num=deconv_num,
                dims=num_dims,
                num_filters=num_filters,
                num_kernels=num_kernels,
                num_strides=num_strides,
                data_format=data_format,
                padding="same",
                kern_reg=space["kernel_reg"],
                act_reg=space["act_reg"],
                bias_reg=space["bias_reg"],
                activation=deconv_act,
                kernel_init=space["kernel_init_dist"],
                bias_init=space["bias_init_dist"],
                trans=True,
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
