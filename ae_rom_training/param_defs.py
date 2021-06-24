from hyperopt import hp
from hyperopt.pyll import scope
from numpy import nan
import numpy as np

# some reference logarithms
# log(-9.21034037) ~= 0.0001
# log(-6.90775527) ~= 0.001,
# log(-4.60517018) ~= 0.01
# log(-2.30258509) ~= 0.1,

# TODO: add optimizer options (just Adam right now)

# list of viable parameters, given as keyword-dtype-default triplets
# if default is nan, then it is a required parameter and has no default
params = [
    ["all_conv", bool, False],  # whether network is all-convolutional
    ["latent_dim", int, nan],  # latent dimension
    ["centering_scheme", str, nan],  # data centering method
    ["normal_scheme", str, nan],  # data normalization scheme
    ["activation_func", str, nan],  # layer activation
    ["final_activation_func", str, nan],  # final layer activation
    ["stride_list", [tuple], nan],  # stride length at each layer
    ["num_conv_layers", int, nan],  # number of convolutional layers
    ["kern_size_fixed", int, nan],  # uniform kernel size
    # specify filter growth behavior, fixed kernel
    ["num_filt_start", int, nan],  # number of filters at initial convolutional layer
    ["filt_growth_mult", int, nan],  # growth rate of number of filters from layer to layer
    ["kernel_reg_type", str, None],  # weight regularization type
    ["kernel_reg_val", float, 0.0],  # weight regularization value
    ["kernel_init_dist", str, "glorot_uniform"],  # initial distribution of kernel values
    ["act_reg_type", str, None],  # activity regularization type
    ["act_reg_val", float, 0.0],  # activity regularization value
    ["bias_reg_type", str, None],  # bias regularization type
    ["bias_reg_val", float, 0.0],  # bias regularization value
    ["bias_init_dist", str, "zeros"],  # initial distribution of biases
    ["learn_rate", float, 1e-4],  # optimization algorithm learning rate
    ["max_epochs", int, nan],  # maximum number of training epochs
    ["val_perc", float, nan],  # percentage of dataset to partition as validation set
    ["loss_func", str, "mse"],  # string for loss function reference
    ["es_patience", int, nan],  # number of iterations before early-stopping kicks in
    ["batch_size", int, nan],  # batch size
    ["layer_precision", int, 32],  # either 64 (for double-precision) or 32 (for single-precision)
]


def set_expression(parameter_name, expression_type, input_list: list):
    """
    Generate HyperOpt expression
    input_list has different interpretation depending on expression_type
    """

    if expression_type == "choice":
        expression = hp.choice(parameter_name, input_list)
    elif expression_type == "uniform":
        assert len(input_list) == 2, "uniform expression only accepts 2 inputs (" + parameter_name + ")"
        expression = hp.uniform(parameter_name, input_list[0], input_list[1])
    elif expression_type == "uniformint":
        assert len(input_list) == 2, "uniformint expression only accepts 2 inputs (" + parameter_name + ")"
        expression = hp.uniformint(parameter_name, input_list[0], input_list[1])
    elif expression_type == "quniform":
        assert len(input_list) == 3, "quniform expression only accepts 3 inputs (" + parameter_name + ")"
        expression = hp.quniform(parameter_name, input_list[0], input_list[1], input_list[2])
    elif expression_type == "quniformint":
        assert len(input_list) == 3, "quniformint expression only accepts 3 inputs (" + parameter_name + ")"
        expression = scope.int(hp.quniform(parameter_name, input_list[0], input_list[1], input_list[2]))
    elif expression_type == "loguniform":
        assert len(input_list) == 2, "loguniform expression only accepts 2 inputs (" + parameter_name + ")"
        expression = hp.loguniform(parameter_name, input_list[0], input_list[1])
    elif expression_type == "qloguniform":
        assert len(input_list) == 3, "qloguniform expression only accepts 3 inputs (" + parameter_name + ")"
        expression = hp.qloguniform(parameter_name, input_list[0], input_list[1], input_list[2])
    else:
        raise ValueError("Invalid or un-implemented HyperOpt expression_type: " + str(expression_type))

    return expression


def define_param_space(input_dict, use_hyper_opt):
    """
    Define architecture and optimization parameters
    """

    space = {}

    for param_pair in params:

        # no input and no default
        if (param_pair[2] is nan) and (param_pair[0] not in input_dict):
            raise ValueError(param_pair[0] + " is a required input")

        # has input value
        if param_pair[0] in input_dict:
            input_val = input_dict[param_pair[0]]

            # using hyperOpt
            if use_hyper_opt:

                # determine expression type, default to "choice"
                expression_type_name = param_pair[0] + "_expType"
                if expression_type_name in input_dict:
                    expression_type = input_dict[expression_type_name]
                else:
                    expression_type = "choice"

                if type(input_val) is not list:
                    assert expression_type == "choice", 'If providing non-list expression parameters, must use "choice"'
                    input_val = [input_val]

                # TODO: make this more general to other list inputs
                if param_pair[0] == "stride_list":
                    if type(input_val[0]) is not list:
                        input_val = [input_val]

                space[param_pair[0]] = set_expression(param_pair[0], expression_type, input_val)

            # not using hyperOpt
            else:

                if type(param_pair[1]) is type:
                    if (input_val is None) and (param_pair[2] is None):
                        pass
                    else:
                        assert type(input_val) is param_pair[1], (
                            "Data type for " + param_pair[0] + " must be " + str(param_pair[1])
                        )
                elif type(param_pair[1]) is list:
                    assert type(input_val) is list
                    # check that all list elements in input_val match expected type
                    assert all(isinstance(x, param_pair[1][0]) for x in input_val)

                space[param_pair[0]] = input_val

        # does not have input value (but has default)
        else:
            input_val = param_pair[2]  # default value

            if use_hyper_opt:
                # HyperOpt requires list arguments
                input_val = [input_val]
                space[param_pair[0]] = set_expression(param_pair[0], "choice", input_val)

            else:
                space[param_pair[0]] = input_val

        # used on following iteration
        if param_pair[0] == "stride_list":
            stride_list = input_val

        # check that all num_conv_layers is LTE to all stride list lengths
        if param_pair[0] == "num_conv_layers":
            if use_hyper_opt:
                if expression_type == "choice":
                    numLayerVals = input_val
                elif expression_type == "quniform":
                    numLayerVals = np.arange(input_val[0], input_val[1], input_val[2])
                elif expression_type == "quniformint":
                    numLayerVals = np.arange(input_val[0], input_val[1], input_val[2])
                else:
                    raise ValueError("Can't check num_conv_layers with expression type " + expression_type)
                assert all(
                    x <= len(y) for x in numLayerVals for y in stride_list
                ), "all values of num_conv_layers must be less than or equal to lengths of all stride_list's"
            else:
                assert input_val <= len(
                    stride_list
                ), "num_conv_layers must be less than or equal to the length of stride_list"

    return space
