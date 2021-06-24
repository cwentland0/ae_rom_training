from tensorflow.keras.layers import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.keras.losses import MeanSquaredError
import tensorflow.keras.backend as K


def transfer_weights(model1, model2):
    """
    Transfer weights from one network to another
    Useful for building explicit-batch network from trained implicit-batch network
    """

    model2.set_weights(model1.get_weights())
    return model2


def get_loss(loss_name):
    """
    Switch function for network loss function
    """

    if loss_name == "pure_l2":
        return pure_l2
    elif loss_name == "pure_mse":
        return pure_mse
    else:
        return loss_name  # assumed to be a built-in loss string


def pure_l2(y_true, y_pred):
    """
    Strict l2 error (opposed to mean-squared)
    """

    return K.sum(K.square(y_true - y_pred))


def pure_mse(y_true, y_pred):
    """
    Strict mean-squared error, not including regularization contribution
    """

    mse = MeanSquaredError()
    return mse(y_true, y_pred)


# defines 1D, 2D, and 3D convolutions, as well as their corresponding transpose convolutions
# N.B.: ASSUMED "channels_first" FORMAT!
def set_conv_layer(
    inputVals,
    conv_num,
    dims,
    num_filters,
    num_kernels,
    num_strides,
    data_format,
    padding="same",
    kern_reg=None,
    act_reg=None,
    bias_reg=None,
    activation="None",
    kernel_init="glorot_uniform",
    bias_init="zeros",
    trans=False,
):

    if trans:
        name = "tconv" + str(conv_num)
        if dims == 1:
            x = Conv1DTranspose(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(inputVals)
        elif dims == 2:
            x = Conv2DTranspose(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(inputVals)
        elif dims == 3:
            x = Conv3DTranspose(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(inputVals)
        else:
            raise ValueError("Invalid dimensions for transpose convolutional layer " + str(conv_num))
    else:
        name = "conv" + str(conv_num)
        if dims == 1:
            x = Conv1D(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(inputVals)
        elif dims == 2:
            x = Conv2D(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(inputVals)
        elif dims == 3:
            x = Conv3D(
                filters=num_filters,
                kernel_size=num_kernels,
                strides=num_strides,
                padding=padding,
                activation=activation,
                kernel_initializer=kernel_init,
                bias_initializer=bias_init,
                kernel_regularizer=kern_reg,
                activity_regularizer=act_reg,
                bias_regularizer=bias_reg,
                data_format=data_format,
                name=name,
            )(inputVals)
        else:
            raise ValueError("Invalid dimensions for convolutional layer " + str(conv_num))

    return x
