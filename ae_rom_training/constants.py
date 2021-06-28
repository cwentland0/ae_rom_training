from numpy import nan

RANDOM_SEED = 24

def process_param_list(param_list):
    param_names = [item[0] for item in param_list]
    param_dtypes = [item[1] for item in param_list]
    param_defaults = [item[2] for item in param_list]

    return param_names, param_dtypes, param_defaults

# some reference logarithms for use with HyperOpt
TENTHOUSANDTH_LOG = -9.21034037 # log(-9.21034037) ~= 0.0001
THOUSANDTH_LOG = -6.90775527 # log(-6.90775527) ~= 0.001,
HUNDREDTH_LOG = -4.60517018 # log(-4.60517018) ~= 0.01
TENTH_LOG = -2.30258509 # log(-2.30258509) ~= 0.1,

# list of viable parameters, given as keyword-dtype-default triplets
# if default is nan, then it is a required parameter and has no default
# layer parameters, should have model prefix in input file
LAYER_PARAM_LIST = [
    ["activation_func", str, nan],  # layer activation
    ["stride", tuple, (1,)],  # stride length
    ["dilation", tuple, (1,)],
    ["padding", str, nan],
    ["kern_size", int, nan],  # uniform kernel size
    ["kernel_reg_type", str, None],  # weight regularization type
    ["kernel_reg_val", float, 0.0],  # weight regularization value
    ["kernel_init_dist", str, "glorot_uniform"],  # initial distribution of kernel values
    ["act_reg_type", str, None],  # activity regularization type
    ["act_reg_val", float, 0.0],  # activity regularization value
    ["bias_reg_type", str, None],  # bias regularization type
    ["bias_reg_val", float, 0.0],  # bias regularization value
    ["bias_init_dist", str, "zeros"],  # initial distribution of biases
    ["target_shape", tuple, nan],  # target shape for reshape layers
]
LAYER_PARAM_NAMES, LAYER_PARAM_DEFAULTS, LAYER_PARAM_DTYPES = process_param_list(LAYER_PARAM_LIST)

# training parameters
TRAIN_PARAM_LIST = [
    ["learn_rate", float, 1e-4],  # optimization algorithm learning rate
    ["max_epochs", int, nan],  # maximum number of training epochs
    ["loss_func", str, "mse"],  # string for loss function reference
    ["es_patience", int, nan],  # number of iterations before early-stopping kicks in
    ["batch_size", int, nan],  # batch size
]
TRAIN_PARAM_NAMES, TRAIN_PARAM_DEFAULTS, TRAIN_PARAM_DTYPES = process_param_list(TRAIN_PARAM_LIST)
