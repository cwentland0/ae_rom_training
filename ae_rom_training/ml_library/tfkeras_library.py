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

    def get_input_layer(self, input_shape, batch_size=None):

        output_layer = Input(shape=input_shape, batch_size=batch_size)
        return output_layer

    def get_conv_layer(
        self,
        layer_input,
        dims,
        num_filters,
        num_kernels,
        num_strides,
        data_format,
        activation="None",
        padding="same",
        kern_reg=None,
        act_reg=None,
        bias_reg=None,
        kern_init="glorot_uniform",
        bias_init="zeros",
    ):
        
        if dims == 1:
            layer_output = Conv1D(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format
            )(layer_input)
        elif dims == 2:
            layer_output = Conv2D(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
            )(layer_input)
        elif dims == 3:
            layer_output = Conv3D(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
            )(layer_input)
        else:
            raise ValueError("Invalid dimensions for convolutional layer: " + str(dims))

        return layer_output

    def get_trans_conv_layer(
        self,
        layer_input,
        dims,
        num_filters,
        num_kernels,
        num_strides,
        data_format,
        activation="None",
        padding="same",
        kern_reg=None,
        act_reg=None,
        bias_reg=None,
        kern_init="glorot_uniform",
        bias_init="zeros",
    ):
        
        if dims == 1:
            layer_output = Conv1DTranspose(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
            )(layer_input)
        elif dims == 2:
            layer_output = Conv2DTranspose(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
            )(layer_input)
        elif dims == 3:
            layer_output = Conv3DTranspose(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kern_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
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
        act_reg=None,
        bias_reg=None,
        kern_init="glorot_uniform",
        bias_init="zeros",
    ):

        # if input is not flattened (batch dimension plus data dimension), flatten and note this change
        added_flatten = False
        if tf.rank(layer_input) > 2:
            layer_input = self.get_flatten_layer(layer_input)
            added_flatten = True

        layer_output = Dense(
            output_size,
            activation=activation,
            kernel_regularizer=kern_reg,
            activity_regularizer=act_reg,
            bias_regularizer=bias_reg,
            kernel_initializer=kern_init,
            bias_initializer=bias_init,
        )(layer_input)
    
        return layer_output, added_flatten

    def get_reshape_layer(self, layer_input, target_shape):
        """"Implement tensor input reshape."""

        output_layer = Reshape(target_shape)(layer_input)
        return output_layer

    def get_flatten_layer(self, layer_input):
        """Implement tensor input flatten."""

        output_layer = Flatten()(layer_input)
        return output_layer

    # return regularizer objects
    def get_regularization(self, reg_name, reg_mult):

        if reg_name == "l2":
            return l2(reg_mult)
        elif reg_name == "l1":
            return l1(reg_mult)
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

    def get_callbacks():
        pass

    def pure_l2(self, y_true, y_pred):
        """Strict L2 error"""

        return K.sum(K.square(y_true - y_pred))


    def pure_mse(y_true, y_pred):
        """Strict mean-squared error, not including regularization contribution"""

        mse = MeanSquaredError()
        return mse(y_true, y_pred)

    def transfer_weights(model1, model2):
        """Transfer weights from one network to another.
        
        Useful for building explicit-batch network from trained implicit-batch network
        """

        model2.set_weights(model1.get_weights())
        return model2