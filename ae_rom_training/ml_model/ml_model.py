from numpy import nan

from ae_rom_training.constants import LAYER_PARAM_NAMES, LAYER_PARAM_DEFAULTS, LAYER_PARAM_DTYPES

class MLModel():

    def __init__(self, input_dict, mllib):
        
        self.mllib = mllib
        self.len_prefix = len(self.param_prefix)

        # layer_list contains all outputs of layer calls
        # layer list should have dicts which have all the parameters needed for a given layer
        # also, should specify which layer (in the list order) is its input
        self.layer_list = []

        self.preproc_inputs(input_dict)

    def preproc_layer_inputs_fixed(self):

        # for a given layer type, check that all required inputs are present

        # loop through all inputs

        # convolution
            # activation_func, stride, kern_size, 

        pass

    def preproc_layer_inputs_hyperopt(self):
        pass

    def get_input_param(self, param_name, input_dict, use_hyper_opt=False):

        param_name_full = self.param_prefix + "_" + param_name

        if param_name in LAYER_PARAM_NAMES:
            param_idx = LAYER_PARAM_NAMES.index(param_name)
            param_dtype = LAYER_PARAM_DTYPES[param_idx]
            param_default = LAYER_PARAM_DEFAULTS[param_idx]

            # no input and no default
            if (param_default is nan) and (param_name_full not in input_dict):
                raise ValueError(param_name_full + " is a required input")

            # has input value
            if param_name_full in input_dict:
                input_val = input_dict[param_name_full]

                # using hyperOpt
                if use_hyper_opt:

                    # determine expression type, default to "choice"
                    expression_type_name = param_name_full + "_expr"
                    if expression_type_name in input_dict:
                        expression_type = input_dict[expression_type_name]
                    else:
                        expression_type = "choice"

                    if type(input_val) is not list:
                        assert expression_type == "choice", 'If providing non-list expression parameters, must use "choice"'
                        input_val = [input_val]

                    space[param_set[0]] = set_expression(param_set[0], expression_type, input_val)

                # not using hyperOpt
                else:

                    if type(param_set[1]) is type:
                        if (input_val is None) and (param_set[2] is None):
                            pass
                        else:
                            assert type(input_val) is param_set[1], (
                                "Data type for " + param_set[0] + " must be " + str(param_set[1])
                            )
                    elif type(param_set[1]) is list:
                        assert type(input_val) is list
                        # check that all list elements in input_val match expected type
                        assert all(isinstance(x, param_set[1][0]) for x in input_val)

                    space[param_set[0]] = input_val

            # does not have input value (but has default)
            else:
                input_val = param_set[2]  # default value

                if use_hyper_opt:
                    # HyperOpt requires list arguments
                    input_val = [input_val]
                    space[param_set[0]] = set_expression(param_set[0], "choice", input_val)

                else:
                    space[param_set[0]] = input_val



        else:
            raise ValueError('No entry for parameter name "' + str(param_name) + '" in ae_rom_training.constants.PARAM_LIST')

        return param_expr

    def assemble(self, input_shape, batch_size=None):

        # start off with input layer
        self.layer_list.append(self.mllib.get_input_layer(input_shape, batch_size=batch_size))

        for layer_dict in self.layer_list:

            # TODO: check that this works with non-sequential topologies
            layer_type = layer_dict["layer_type"]
            layer_input = self.layer_list[layer_dict["layer_input_idx"]]

            # convolution layer
            if layer_type == "conv":
                layer_output = self.mllib.get_conv_layer(
                    layer_input,
                    layer_dict["dims"],
                    layer_dict["num_filters"],
                    layer_dict["num_kernels"],
                    layer_dict["num_strides"],
                    layer_dict["data_format"],
                    activation=layer_dict["activation"],
                    padding=layer_dict["padding"],
                    kern_reg=layer_dict["kern_reg"],
                    act_reg=layer_dict["act_reg"],
                    bias_reg=layer_dict["bias_reg"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                )

            # transpose convolution layer
            if layer_type == "trans_conv":
                layer_output = self.mllib.get_trans_conv(
                    layer_input,
                    layer_dict["dims"],
                    layer_dict["num_filters"],
                    layer_dict["num_kernels"],
                    layer_dict["num_strides"],
                    layer_dict["data_format"],
                    activation=layer_dict["activation"],
                    padding=layer_dict["padding"],
                    kern_reg=layer_dict["kern_reg"],
                    act_reg=layer_dict["act_reg"],
                    bias_reg=layer_dict["bias_reg"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                )

            # dense/fully-connected layer
            elif layer_type == "dense":
                layer_output, added_flatten = self.mllib.get_dense_layer(
                    layer_input,
                    layer_dict["output_size"],
                    activation=layer_dict["activation"],
                    kern_reg=layer_dict["kern_reg"],
                    act_reg=layer_dict["act_reg"],
                    bias_reg=layer_dict["bias_reg"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                )
                if added_flatten:
                    self.num_layers += 1

            # reshape layer
            elif layer_type == "reshape":
                layer_output = self.mllib.get_reshape_layer(
                    layer_input,
                    layer_dict["target_shape"],
                )

            # flatten layer
            elif layer_type == "flatten":
                layer_output = self.mllib.get_flatten_layer(
                    layer_input,
                )

            else:
                raise 

            self.layer_list.append(layer_output)


    def train(self):

        # get optimizer
        # get loss
        # get callbacks
        # train
        pass

        

    def save(self, out_dir):
        pass


