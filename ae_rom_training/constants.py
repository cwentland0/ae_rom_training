from numpy import nan

RANDOM_SEED = 24
TRAIN_VERBOSITY = "min"

# some reference logarithms for use with HyperOpt
TENTHOUSANDTH_LOG = -9.21034037  # log(-9.21034037) ~= 0.0001
THOUSANDTH_LOG = -6.90775527  # log(-6.90775527) ~= 0.001,
HUNDREDTH_LOG = -4.60517018  # log(-4.60517018) ~= 0.01
TENTH_LOG = -2.30258509  # log(-2.30258509) ~= 0.1,

# dict of viable layer parameters, given as dtype-default doublets
# if default is nan, then it is a required parameter and has no default
# layer parameters should have model prefix in input file
LAYER_PARAM_DICT = {
    "layer_type": [str, nan],
    "layer_input_idx": [int, -1],
    "activation": [str, nan],  # layer activation
    "num_filters": [int, nan],
    "strides": [(int,), nan],  # stride length for convolutional layers
    "dilation": [(int,), nan],  # dilation size for convolutional layers
    "padding": [str, "same"],  # padding type for convolutional layers
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
    "use_bias": [bool, True]  # whether to apply bias to a layer's calculations
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
    "alpha": [float, 0.5],  # weighting factor for autoencoder reconstruction error in combined loss
    "beta": [float, 0.5],  # weighting factor for time-stepper error in combined loss
    "eps": [float, 1e-12],  # additive factor to prevent denominator from becoming zero
}
