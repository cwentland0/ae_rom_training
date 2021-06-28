import os

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import Flatten, Reshape
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K

from ae_rom_training.constants import RANDOM_SEED
from ae_rom_training.ml_library.ml_library import MLLibrary


class TFKerasLibrary(MLLibrary):
    """Functionality for Tensorflow-Keras"""

    def __init__(self, input_dict):

        tf.random.set_seed(RANDOM_SEED)
        super().__init__(input_dict)

    def assemble_model(self):
        pass

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
        num_strides,
        conv_order,
        activation="None",
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
        
        if conv_order == "NCHW":
            data_format = "channels_first"
        elif conv_order == "NHWC":
            data_format = "channels_last"
        else:
            raise ValueError("Invalid conv_order: " + str(conv_order))

        # set layer
        if dims == 1:
            layer_output = Conv1D(
                filters=num_filters,
                kernel_size=kern_size,
                strides=num_strides,
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
                strides=num_strides,
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
                strides=num_strides,
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
        num_strides,
        conv_order,
        activation="None",
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
        
        if conv_order == "NCHW":
            data_format = "channels_first"
        elif conv_order == "NHWC":
            data_format = "channels_last"
        else:
            raise ValueError("Invalid conv_order: " + str(conv_order))

        # set layer
        if dims == 1:
            layer_output = Conv1DTranspose(
                filters=num_filters,
                kernel_size=kern_size,
                strides=num_strides,
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
                strides=num_strides,
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
                strides=num_strides,
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
        activation="None",
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
            kernel_regularizer=kern_reg,
            activity_regularizer=act_reg,
            bias_regularizer=bias_reg,
            kernel_initializer=kern_init,
            bias_initializer=bias_init,
            name=name,
        )(layer_input)
    
        return layer_output, added_flatten

    def get_reshape_layer(self, layer_input, target_shape, name=None):
        """"Implement tensor input reshape."""

        output_layer = Reshape(target_shape, name=name)(layer_input)
        return output_layer

    def get_flatten_layer(self, layer_input, name=None):
        """Implement tensor input flatten."""

        output_layer = Flatten(name=name)(layer_input)
        return output_layer

    # return regularizer objects
    def get_regularization(self, reg_name, reg_val):

        if reg_name == "l2":
            return l2(reg_val)
        elif reg_name == "l1":
            return l1(reg_val)
        else:
            raise ValueError("Invalid regularization name: " + str(reg_name))

    def get_optimizer(self, optimizer_name, learning_rate=0.001):
        
        if optimizer_name == "Adam":
            return Adam(learning_rate=learning_rate)
        else:
            raise ValueError("Invalid regularization name: " + str(optimizer_name))

    def get_loss_function(self, loss_name):

        if loss_name == "pure_l2":
            return self.pure_l2
        elif loss_name == "pure_mse":
            return self.pure_mse
        else:
            return loss_name  # assumed to be a built-in loss string

    def get_callbacks(self):
        pass

    def pure_l2(self, y_true, y_pred):
        """Strict L2 error"""

        return K.sum(K.square(y_true - y_pred))


    def pure_mse(self, y_true, y_pred):
        """Strict mean-squared error, not including regularization contribution"""

        mse = MeanSquaredError()
        return mse(y_true, y_pred)

    def transfer_weights(self, model1, model2):
        """Transfer weights from one network to another.
        
        Useful for building explicit-batch network from trained implicit-batch network
        """

        model2.set_weights(model1.get_weights())
        return model2

    def get_tensor_dims(self, tensor):
        return len(tensor.shape.as_list())

    def build_model_obj(self, layer_list):

        return Model(layer_list[0], layer_list[-1])