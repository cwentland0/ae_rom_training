import os

import numpy as np
from numpy import inf
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import LSTM, GRU
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization, LayerNormalization
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Progbar
from tensorflow.keras.models import load_model
from tcn import TCN

from ae_rom_training.constants import RANDOM_SEED, TRAIN_VERBOSITY
from ae_rom_training.ml_library.ml_library import MLLibrary
from ae_rom_training.ml_library.tfkeras.tfkeras_losses import pure_l2, pure_mse, ae_ts_combined_error
from ae_rom_training.ml_library.tfkeras.tfkeras_layers import ContinuousKoopman
from ae_rom_training.preproc_utils import get_shape_tuple


if TRAIN_VERBOSITY == "none":
    verbose = 0
elif TRAIN_VERBOSITY == "min":
    verbose = 2
elif TRAIN_VERBOSITY == "max":
    verbose = 1
else:
    raise ValueError("Invalid entry for TRAIN_VERBOSITY: " + str(TRAIN_VERBOSITY))


class TFKerasLibrary(MLLibrary):
    """Functionality for Tensorflow-Keras"""

    def __init__(self, run_gpu=False):

        super().__init__(run_gpu)

    def seed_rng(self):

        tf.random.set_seed(RANDOM_SEED)

    def init_gpu(self, run_gpu):

        # run on CPU vs GPU
        if run_gpu:
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
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    def get_input_layer(self, input_shape, batch_size=None, name=None):

        output_layer = Input(shape=input_shape, batch_size=batch_size, name=name)
        return output_layer

    def get_conv_layer(
        self,
        layer_input,
        dims,
        num_filters,
        kern_size,
        strides,
        dilation,
        network_order,
        activation,
        padding="same",
        kern_reg=None,
        kern_reg_val=0.0,
        act_reg=None,
        act_reg_val=0.0,
        bias_reg=None,
        bias_reg_val=0.0,
        kern_init="glorot_uniform",
        bias_init="zeros",
        name=None,
    ):

        # get regularizers, if requested
        if kern_reg is not None:
            kern_reg = self.get_regularization(kern_reg, kern_reg_val)
        if act_reg is not None:
            act_reg = self.get_regularization(act_reg, act_reg_val)
        if bias_reg is not None:
            bias_reg = self.get_regularization(bias_reg, bias_reg_val)

        if network_order == "NCHW":
            data_format = "channels_first"
        elif network_order == "NHWC":
            data_format = "channels_last"
        else:
            raise ValueError("Invalid network_order: " + str(network_order))

        # set layer
        if dims == 1:
            layer_output = Conv1D(
                filters=num_filters,
                kernel_size=kern_size,
                strides=strides,
                dilation_rate=dilation,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(layer_input)
        elif dims == 2:
            layer_output = Conv2D(
                filters=num_filters,
                kernel_size=kern_size,
                strides=strides,
                dilation_rate=dilation,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(layer_input)
        elif dims == 3:
            layer_output = Conv3D(
                filters=num_filters,
                kernel_size=kern_size,
                strides=strides,
                dilation_rate=dilation,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(layer_input)
        else:
            raise ValueError("Invalid dimensions for convolutional layer: " + str(dims))

        return layer_output

    def get_trans_conv_layer(
        self,
        layer_input,
        dims,
        num_filters,
        kern_size,
        strides,
        dilation,
        network_order,
        activation,
        padding="same",
        kern_reg=None,
        kern_reg_val=0.0,
        act_reg=None,
        act_reg_val=0.0,
        bias_reg=None,
        bias_reg_val=0.0,
        kern_init="glorot_uniform",
        bias_init="zeros",
        name=None,
    ):

        # get regularizers, if requested
        if kern_reg is not None:
            kern_reg = self.get_regularization(kern_reg, kern_reg_val)
        if act_reg is not None:
            act_reg = self.get_regularization(act_reg, act_reg_val)
        if bias_reg is not None:
            bias_reg = self.get_regularization(bias_reg, bias_reg_val)

        if network_order == "NCHW":
            data_format = "channels_first"
        elif network_order == "NHWC":
            data_format = "channels_last"
        else:
            raise ValueError("Invalid network_order: " + str(network_order))

        # set layer
        if dims == 1:
            layer_output = Conv1DTranspose(
                filters=num_filters,
                kernel_size=kern_size,
                strides=strides,
                dilation_rate=dilation,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(layer_input)
        elif dims == 2:
            layer_output = Conv2DTranspose(
                filters=num_filters,
                kernel_size=kern_size,
                strides=strides,
                dilation_rate=dilation,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(layer_input)
        elif dims == 3:
            layer_output = Conv3DTranspose(
                filters=num_filters,
                kernel_size=kern_size,
                strides=strides,
                dilation_rate=dilation,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(layer_input)
        else:
            raise ValueError("Invalid dimensions for transpose convolutional layer: " + str(dims))

        return layer_output

    def get_dense_layer(
        self,
        layer_input,
        output_size,
        activation,
        use_bias=True,
        kern_reg=None,
        kern_reg_val=0.0,
        act_reg=None,
        act_reg_val=0.0,
        bias_reg=None,
        bias_reg_val=0.0,
        kern_init="glorot_uniform",
        bias_init="zeros",
        name=None,
        flatten_count=None,
    ):

        # get regularizers, if requested
        if kern_reg is not None:
            kern_reg = self.get_regularization(kern_reg, kern_reg_val)
        if act_reg is not None:
            act_reg = self.get_regularization(act_reg, act_reg_val)
        if bias_reg is not None:
            bias_reg = self.get_regularization(bias_reg, bias_reg_val)

        # if input is not flattened (batch dimension plus data dimension), flatten and note this change
        added_flatten = False
        if self.get_tensor_dims(layer_input) > 2:
            if flatten_count is not None:
                flatten_name = "flatten_" + str(flatten_count)
            else:
                flatten_name = None
            layer_input = self.get_flatten_layer(layer_input, name=flatten_name)
            added_flatten = True

        layer_output = Dense(
            output_size,
            activation=activation,
            use_bias=use_bias,
            kernel_regularizer=kern_reg,
            activity_regularizer=act_reg,
            bias_regularizer=bias_reg,
            kernel_initializer=kern_init,
            bias_initializer=bias_init,
            name=name,
        )(layer_input)

        return layer_output, added_flatten

    def get_continuous_koopman_layer(
        self, layer_input, output_size, num_input_layers, stable=False, kern_init="glorot_uniform", name=None,
    ):

        time_input = self.get_input_layer((1,), name="input_" + str(num_input_layers))

        layer_output = ContinuousKoopman(output_size, stable=stable, kernel_initializer=kern_init, name=name,)(
            [layer_input, time_input]
        )

        return [time_input, layer_output]

    def get_lstm_layer(
        self,
        layer_input,
        output_size,
        return_sequences,
        activation,
        recurrent_activation="sigmoid",
        use_bias=True,
        kern_reg=None,
        kern_reg_val=0.0,
        act_reg=None,
        act_reg_val=0.0,
        bias_reg=None,
        bias_reg_val=0.0,
        recurrent_reg=None,
        recurrent_reg_val=0.0,
        kern_init="glorot_uniform",
        bias_init="zeros",
        recurrent_init="orthogonal",
        name=None,
    ):

        # get regularizers, if requested
        if kern_reg is not None:
            kern_reg = self.get_regularization(kern_reg, kern_reg_val)
        if act_reg is not None:
            act_reg = self.get_regularization(act_reg, act_reg_val)
        if bias_reg is not None:
            bias_reg = self.get_regularization(bias_reg, bias_reg_val)
        if recurrent_reg is not None:
            recurrent_reg = self.get_regularization(recurrent_reg, recurrent_reg_val)

        layer_output = LSTM(
            output_size,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kern_init,
            recurrent_initializer=recurrent_init,
            bias_initializer=bias_init,
            kernel_regularizer=kern_reg,
            recurrent_regularizer=recurrent_reg,
            return_sequences=return_sequences,
            name=name,
        )(layer_input)

        return layer_output

    def get_gru_layer(
        self,
        layer_input,
        output_size,
        return_sequences,
        activation,
        recurrent_activation="sigmoid",
        use_bias=True,
        kern_reg=None,
        kern_reg_val=0.0,
        act_reg=None,
        act_reg_val=0.0,
        bias_reg=None,
        bias_reg_val=0.0,
        recurrent_reg=None,
        recurrent_reg_val=0.0,
        kern_init="glorot_uniform",
        bias_init="zeros",
        recurrent_init="orthogonal",
        name=None,
    ):

        # get regularizers, if requested
        if kern_reg is not None:
            kern_reg = self.get_regularization(kern_reg, kern_reg_val)
        if act_reg is not None:
            act_reg = self.get_regularization(act_reg, act_reg_val)
        if bias_reg is not None:
            bias_reg = self.get_regularization(bias_reg, bias_reg_val)
        if recurrent_reg is not None:
            recurrent_reg = self.get_regularization(recurrent_reg, recurrent_reg_val)

        layer_output = GRU(
            output_size,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kern_init,
            recurrent_initializer=recurrent_init,
            bias_initializer=bias_init,
            kernel_regularizer=kern_reg,
            recurrent_regularizer=recurrent_reg,
            return_sequences=return_sequences,
            name=name,
        )(layer_input)

        return layer_output

    def get_tcn_layer(
        self,
        layer_input,
        num_filters,
        kern_size,
        activation,
        dilations,
        return_sequences,
        padding="causal",
        kern_init="glorot_uniform",
        name=None,
    ):

        layer_output = TCN(
            nb_filters=num_filters,
            kernel_size=kern_size,
            dilations=dilations,
            padding=padding,
            return_sequences=return_sequences,
            activation=activation,
            kernel_initializer=kern_init,
            name=name,
        )(layer_input)

        return layer_output

    def get_dropout_layer(self, layer_input, dropout_rate, name=None):

        output_layer = Dropout(dropout_rate, seed=RANDOM_SEED, name=name)(layer_input)
        return output_layer

    def get_batchnorm_layer(self, layer_input, network_order, name=None):

        input_rank = len(layer_input.shape) - 1  # exclude batch dimension
        input_name = layer_input._name
        if (
            (input_name[:6] in ["conv1d", "conv2d", "conv3d"])
            or (input_name[:11] in ["transconv1d", "transconv2d", "transconv3d"])
        ):
            if network_order == "NCHW":
                axis = 1
            elif network_order == "NHWC":
                axis = -1
            else:
                raise ValueError("Invalid network_order: " + str(network_order))
        elif input_name[:5] == "dense":
            axis = 1
        elif (input_name[:7] in ["dropout", "reshape"]):
            # definitely not appropriate here, oh well
            if input_rank == 1:
                axis = 1
            elif input_rank > 1:
                if network_order == "NCHW":
                    axis = 1
                elif network_order == "NHWC":
                    axis = -1
                else:
                    raise ValueError("Invalid network_order: " + str(network_order))
            else:
                raise ValueError("Input to batchnorm rank is non-positive?")
        else:
            raise ValueError("Failed to place batchnorm after unexpected layer: " + input_name)

        # TODO: add other options if it becomes relevant
        output_layer = BatchNormalization(axis=axis, name=name)(layer_input)
        return output_layer

    def get_layernorm_layer(self, layer_input, network_order, name=None):

        input_rank = len(layer_input.shape) - 1  # exclude batch dimension
        input_name = layer_input._name
        if (
            (input_name[:6] in ["conv1d", "conv2d", "conv3d"])
            or (input_name[:11] in ["transconv1d", "transconv2d", "transconv3d"])
        ):
            if network_order == "NCHW":
                axis = 1
            elif network_order == "NHWC":
                axis = -1
            else:
                raise ValueError("Invalid network_order: " + str(network_order))
        elif input_name[:5] == "dense":
            axis = 1
        elif (input_name[:7] in ["dropout", "reshape"]):
            # definitely not appropriate here, oh well
            if input_rank == 1:
                axis = 1
            elif input_rank > 1:
                if network_order == "NCHW":
                    axis = 1
                elif network_order == "NHWC":
                    axis = -1
                else:
                    raise ValueError("Invalid network_order: " + str(network_order))
            else:
                raise ValueError("Input to batchnorm rank is non-positive?")
        else:
            raise ValueError("Failed to place batchnorm after unexpected layer: " + input_name)

        # TODO: add other options if it becomes relevant
        output_layer = LayerNormalization(axis=axis, name=name)(layer_input)
        return output_layer

    def get_reshape_layer(self, layer_input, target_shape, name=None):
        """"Implement tensor input reshape."""

        output_layer = Reshape(target_shape, name=name)(layer_input)
        return output_layer

    def get_flatten_layer(self, layer_input, name=None):
        """Implement tensor input flatten."""

        output_layer = Flatten(name=name)(layer_input)
        return output_layer

    def get_regularization(self, reg_name, reg_val):

        if reg_name == "l2":
            return l2(reg_val)
        elif reg_name == "l1":
            return l1(reg_val)
        else:
            raise ValueError("Invalid regularization name: " + str(reg_name))

    def transfer_weights(self, model1, model2):
        """Transfer weights from one network to another.

        Useful for building explicit-batch network from trained implicit-batch network
        """

        model2.set_weights(model1.get_weights())
        return model2

    def get_layer_weights(self, model, layer_num, weights=True, bias=False):

        # TODO: this is assuming a lot about the form of the weights,
        #   not sure if other layers can have other weight orderings
        #   Might want to check Layer type first and handle accordingly

        if weights:
            if bias:
                return model.layers[layer_num].get_weights()
            else:
                return model.layers[layer_num].get_weights()[0]
        elif bias:
            return model.layers[layer_num].get_weights()[1]
        else:
            raise ValueError("Must set either weights=True or bias=True")

    def get_tensor_dims(self, tensor):
        return len(tensor.shape.as_list())

    def get_layer(self, model_obj, layer_idx):

        return model_obj.layers[layer_idx]

    def get_layer_io_shape(self, model_obj, layer_idx):

        input_shapes = get_shape_tuple(model_obj.layers[layer_idx].input_shape)
        output_shapes = get_shape_tuple(model_obj.layers[layer_idx].output_shape)

        # strip batch dimension, deal with multiple input layers
        if isinstance(input_shapes, list):
            input_shapes = [shape[1:] for shape in input_shapes]
        else:
            input_shapes = input_shapes[1:]

        if isinstance(output_shapes, list):
            output_shapes = [shape[1:] for shape in output_shapes]
        else:
            output_shapes = output_shapes[1:]

        return input_shapes, output_shapes

    def build_model_obj(self, tensor_list, input_idx_list):

        # TODO: handle multiple outputs

        input_list = []
        for input_idx in input_idx_list:
            input_list.append(tensor_list[input_idx])

        model = Model(input_list, tensor_list[-1])

        return model

    def display_model_summary(self, model_obj, displaystr=None):

        if displaystr is not None:
            print(displaystr)
        model_obj.summary()

    def merge_models(self, model_obj_list):

        input_tensor = model_obj_list[0].layers[0].input
        output_tensor = input_tensor
        for model in model_obj_list:
            output_tensor = model(output_tensor)

        return Model(input_tensor, output_tensor)

    def get_model_layer_type_list(self, model_obj):

        type_list = []
        for layer in model_obj.layers:
            layer_type = layer.__class__.__name__
            if layer_type == "InputLayer":
                type_list.append("input")
            elif layer_type[:4] == "Conv":
                type_list.append(layer_type.lower())
            elif layer_type == "Dense":
                type_list.append("dense")
            elif layer_type == "Flatten":
                type_list.append("flatten")
            elif layer_type == "Reshape":
                type_list.append("reshape")

        return type_list

    def get_optimizer(self, params):

        optimizer_name = params["optimizer"]
        if optimizer_name == "Adam":
            return Adam(learning_rate=params["learn_rate"])
        else:
            raise ValueError("Invalid regularization name: " + str(optimizer_name))

    def get_loss_function(self, params):

        loss_name = params["loss_func"]
        if loss_name == "pure_l2":
            return pure_l2
        elif loss_name == "pure_mse":
            return pure_mse
        elif loss_name == "ae_ts_combined":
            return ae_ts_combined_error
        else:
            return loss_name  # assumed to be a built-in loss string

    def get_options(self, params):

        callback_list = []
        added_callback = False

        # early stopping
        if params["early_stopping"]:
            es_patience = params["es_patience"]
            early_stop = EarlyStopping(patience=int(es_patience), restore_best_weights=True)
            callback_list.append(early_stop)
            added_callback = True

        options = {}
        if added_callback:
            options["callbacks"] = callback_list
        else:
            options["callbacks"] = None

        return options

    def train_model_builtin(
        self,
        model_obj,
        data_input_train,
        data_output_train,
        data_input_val,
        data_output_val,
        optimizer,
        loss,
        options,
        input_dict,
        params,
    ):

        model_obj.compile(optimizer=optimizer, loss=loss)

        batch_size = params["batch_size"]
        max_epochs = params["max_epochs"]
        history = model_obj.fit(
            x=data_input_train,
            y=data_output_train,
            batch_size=int(batch_size),
            epochs=int(max_epochs),
            validation_data=(data_input_val, data_output_val),
            verbose=verbose,
            callbacks=options["callbacks"],
        )

        # report training and validation loss
        loss_train = self.calc_loss(model_obj, data_input_train, data_output_train)
        loss_val = self.calc_loss(model_obj, data_input_val, data_output_val)
        loss_train_hist = np.array(history.history["loss"])
        loss_val_hist = np.array(history.history["val_loss"])

        return loss_train_hist, loss_val_hist, loss_train, loss_val

    def train_model_custom(
        self,
        ae_rom,
        data_input_train,
        data_output_train,
        data_input_val,
        data_output_val,
        optimizer,
        loss,
        options,
        params,
        continuous=False,
        time_values_train=None,
        time_values_val=None,
        **kwargs,
    ):

        # TODO: could roll time values into data_input_train, make it a dict?

        batch_size = params["batch_size"]
        max_epochs = params["max_epochs"]
        loss_train_hist = np.zeros(max_epochs)
        loss_val_hist = np.zeros(max_epochs)
        loss_addtl_train_list = []
        loss_addtl_val_list = []

        # deal with needing time values for continuous models
        if not continuous:
            continuous = False
            time_values_train_batch = None

        # get batch iterator
        # assumed that leading dimension of data is batch dimension
        num_samps = data_input_train.shape[0]
        batch_zip = [i for i in zip(range(0, num_samps, batch_size), range(batch_size, num_samps + 1, batch_size))]
        if batch_zip[-1][-1] != num_samps:
            if batch_zip[-1][-1] == (num_samps - 1):
                batch_zip[-1] = (batch_zip[-1][0], num_samps)
            else:
                batch_zip.append((batch_zip[-1][-1], num_samps))

        # get trainable variables
        trainable_vars = []
        trainable_vars_ae = []
        for grad_idx, grad_source in enumerate(ae_rom.grad_source_list):
            trainable_vars += grad_source.trainable_variables
            if grad_idx != 2:
                trainable_vars_ae += grad_source.trainable_variables

        metrics = ["loss", "val_loss"]

        # early stopping
        # TODO: this is only valid for val_loss that is expected to decrease with training
        es = None
        if isinstance(options["callbacks"], list):
            for callback in options["callbacks"]:
                if isinstance(callback, EarlyStopping):
                    es = callback
                    break
            if es is not None:
                es_epochs = 0
                es_loss = inf

        # training loop
        check_ae_loss = np.Inf
        start_full = False
        for epoch in range(max_epochs):
            print("Epoch " + str(epoch + 1) + "/" + str(max_epochs))

            progbar = Progbar(num_samps, stateful_metrics=metrics, verbose=verbose)

            if (check_ae_loss > 0.01) and (not start_full):
                only_ae = True
                trainable_vars_in = trainable_vars_ae
            else:
                start_full = True
                only_ae = False
                trainable_vars_in = trainable_vars

            # batch loop
            loss_val = 0.0
            for batch, (start, end) in enumerate(batch_zip):
                data_input_train_batch = data_input_train[start:end, ...]
                data_output_train_batch = data_output_train[start:end, ...]

                if continuous:
                    time_values_train_batch = time_values_train[start:end, ...]

                # compute gradients and back-propagate
                with tf.GradientTape() as tape:
                    loss_train_list = loss(
                        data_input_train_batch,
                        data_output_train_batch,
                        ae_rom,
                        continuous=continuous,
                        time_values=time_values_train_batch,
                        eps=params["eps"],
                        only_ae=only_ae,
                        **kwargs,
                    )
                    loss_train = loss_train_list[0]
                grad = tape.gradient(target=loss_train, sources=trainable_vars_in)
                optimizer.apply_gradients(zip(grad, trainable_vars_in))

                if verbose == 1:
                    progbar.update(num_samps, values=[("loss", loss_train.numpy()), ("val_loss", loss_val.numpy()), ()])

            check_ae_loss = loss_train_list[1].numpy()
            loss_val_list = loss(
                data_input_val, data_output_val, ae_rom, continuous=continuous, time_values=time_values_val, only_ae=only_ae, **kwargs,
            )
            loss_val = loss_val_list[0]
            progbar.update(num_samps, values=[("loss", loss_train.numpy()), ("val_loss", loss_val.numpy())])

            # store total loss history
            # NOTE: training loss is only for last batch, way too expensive to re-evaluate entire training set
            loss_train_hist[epoch] = loss_train.numpy()
            loss_val_hist[epoch] = loss_val.numpy()

            # store additional loss histories
            if len(loss_train_list) > 1:
                if epoch == 0:
                    for loss_idx in range(len(loss_train_list) - 1):
                        loss_addtl_train_list.append(np.zeros(max_epochs))
                        loss_addtl_val_list.append(np.zeros(max_epochs))

                if only_ae:
                    loss_train_in = loss_train_list[:-1]
                    loss_val_in = loss_val_list[:-1]
                else:
                    loss_train_in = loss_train_list
                    loss_val_in = loss_val_list
                for loss_idx in range(len(loss_train_in) - 1):
                    loss_addtl_train_list[loss_idx][epoch] = loss_train_in[loss_idx + 1].numpy()
                    loss_addtl_val_list[loss_idx][epoch] = loss_val_in[loss_idx + 1].numpy()

            # check for early stopping
            if es is not None:
                # validation loss has decreased
                if loss_val.numpy() < es_loss:

                    es_epochs = 0
                    es_loss = loss_val.numpy()
                    best_encoder_weights = ae_rom.autoencoder.encoder.model_obj.get_weights()
                    best_decoder_weights = ae_rom.autoencoder.decoder.model_obj.get_weights()
                    # TODO: this can't handle continuous Koopman
                    best_stepper_weights = ae_rom.time_stepper.stepper.model_obj.get_weights()

                # validation loss has NOT decreased
                else:
                    es_epochs += 1

                    # exceeded patience
                    if es_epochs == es.patience:
                        # restore weights
                        if es.restore_best_weights:
                            ae_rom.autoencoder.encoder.model_obj.set_weights(best_encoder_weights)
                            ae_rom.autoencoder.decoder.model_obj.set_weights(best_decoder_weights)
                            ae_rom.time_stepper.stepper.model_obj.set_weights(best_stepper_weights)

                        # truncate loss histories
                        loss_train_hist = loss_train_hist[:epoch]
                        loss_val_hist = loss_val_hist[:epoch]
                        for loss_idx in range(len(loss_train_list) - 1):
                            loss_addtl_train_list[loss_idx] = loss_addtl_train_list[loss_idx][:epoch]
                            loss_addtl_val_list[loss_idx] = loss_addtl_val_list[loss_idx][:epoch]

                        # exit epoch loop
                        break

        # calculate final training and validation loss
        loss_train_list = loss(
            data_input_train, data_output_train, ae_rom, continuous=continuous, time_values=time_values_val, **kwargs,
        )
        loss_train = loss_train_list[0].numpy()
        loss_val_list = loss(
            data_input_val, data_output_val, ae_rom, continuous=continuous, time_values=time_values_val, **kwargs,
        )
        loss_val = loss_val_list[0].numpy()

        return loss_train, loss_val, loss_train_hist, loss_val_hist, loss_addtl_train_list, loss_addtl_val_list

    def calc_loss(self, model_obj, input_data, output_data):

        loss = model_obj.evaluate(x=input_data, y=output_data, verbose=0)
        return loss

    def get_koopman(self, model_obj):
        """Compatability layer to get Koopman operator from continuous Koopman model.

        model_obj is assumed to have three layers: Two Input (latent variables and time) and ContinuousKoopman.
        Retrieval of K is handled within custom layer.
        """

        return model_obj.layers[-1].get_koopman_numpy()

    def eval_model(self, model_obj, input_data):

        return model_obj(input_data).numpy()

    def load_model(self, model_dir, model_name):

        # TODO: add ability to load compilable model for resuming training
        # TODO: load from SavedModel

        # check if h5 exists
        h5_path = os.path.join(model_dir, model_name + ".h5")
        if os.path.isfile(h5_path):
            model_obj = load_model(h5_path, compile=False)

        else:
            raise ValueError("Could not find h5 model at " + h5_path)

        return model_obj

    def save_model(self, model_obj, save_path, save_h5=True):
        """Save Tensorflow model object.

        save_path should NOT have a file extension, as this is appended based on save_h5.
        """

        _, ext = os.path.splitext(save_path)
        assert not ext, "save_path should not have a file extension; it currently has: " + ext

        if save_h5:
            model_obj.save(save_path + ".h5", save_format="h5")
        else:
            model_obj.save(save_path)
