from numpy import nan

RANDOM_SEED = 24
TRAIN_VERBOSITY = "min"

# some reference logarithms for use with HyperOpt
TENTHOUSANDTH_LOG = -9.21034037  # log(-9.21034037) ~= 0.0001
THOUSANDTH_LOG = -6.90775527  # log(-6.90775527) ~= 0.001,
HUNDREDTH_LOG = -4.60517018  # log(-4.60517018) ~= 0.01
TENTH_LOG = -2.30258509  # log(-2.30258509) ~= 0.1,

# list of viable parameters, given as keyword-dtype-default triplets
# if default is nan, then it is a required parameter and has no default
# layer parameters, should have model prefix in input file
LAYER_PARAM_DICT = {
    "layer_type": [str, nan],
    "layer_input_idx": [int, nan],
    "activation": [str, nan],  # layer activation
    "num_filters": [int, nan],
    "strides": [(int,), (1,)],  # stride length for convolutional layers
    "dilation": [(int,), (1,)],  # dilation size for convolutional layers
    "padding": [str, nan],  # padding type for convolutional layers
    "kern_size": [int, nan],  # uniform kernel size
    "kern_reg": [str, None],  # weight regularization type
    "kern_reg_val": [float, 0.0],  # weight regularization value
    "kern_init": [str, "glorot_uniform"],  # initial distribution of kernel values
    "act_reg": [str, None],  # activity regularization type
    "act_reg_val": [float, 0.0],  # activity regularization value
    "bias_reg": [str, None],  # bias regularization type
    "bias_reg_val": [float, 0.0],  # bias regularization value
    "bias_init": [str, "zeros"],  # initial distribution of biases
    "output_size": [int, nan],  # output size for dense layers
    "target_shape": [(int,), nan],  # target shape for reshape layers
}

# training parameters
TRAIN_PARAM_DICT = {
    "learn_rate": [float, 1e-4],  # optimization algorithm learning rate
    "max_epochs": [int, nan],  # maximum number of training epochs
    "loss_func": [str, "mse"],  # string for loss function reference
    "optimizer": [str, "Adam"],  # training optimizer
    "early_stopping": [bool, False],  # whether to use early stopping
    "es_patience": [int, nan],  # number of iterations before early-stopping kicks in
    "batch_size": [int, nan],  # batch size
}
