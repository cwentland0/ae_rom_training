from numpy import nan
from tensorflow.python.keras.backend import flatten, reshape

from ae_rom_training.constants import LAYER_PARAM_NAMES, LAYER_PARAM_DEFAULTS, LAYER_PARAM_DTYPES

# TODO: handle different network definitions for separate networks

class MLModel():
    """Base class for component ML models that make up the autoencoder.
    
    Examples include the encoder and decoder, as well as time steppers and parameter predictors.

    Args:
        input_dict: dict of input values from input text file.
        param_space: parameter space of containing Autoencoder object, which may contain HyperOpt expressions
        mllib: MLLibrary object defining machine learning library functionalities
    """

    def __init__(self, input_dict, param_space, mllib):
        
        self.mllib = mllib
        self.len_prefix = len(self.param_prefix)

        # layer_list contains all outputs of layer calls
        # layer list should have dicts which have all the parameters needed for a given layer
        # also, should specify which layer (in the list order) is its input
        self.layer_list = []
        self.layer_params_list = []

        self.preproc_inputs(input_dict, param_space)

    def preproc_inputs(self, input_dict, param_space):
        """Reads all input values for a given model type, does preprocessing and simple error checking.
        
        Handles varied situations for using HyperOpt vs. no HyperOpt, list inputs vs. single inputs.
        This is called BEFORE HyperOpt populates the parameter space, in-depth error checking is done later.
        """

        # don't assign a unique parameter space for a mirrored decoder
        if (self.param_prefix == "decoder") and input_dict["mirrored_decoder"]:
            return

        self.input_keys = [key for key, _ in input_dict.items() if key.startswith(self.param_prefix + "_")]
        self.num_layers = len(input_dict[self.param_prefix + "_layer_type"]) # excluding input layer

        # loop through all inputs
        if input_dict["use_hyperopt"]:
            uses_hyperopt_dict = {}

        for input_key in self.input_keys:
            input_value = input_dict[input_key]

            if input_dict["use_hyperopt"]:
                raise ValueError("HyperOpt preprocessing not implemented yet")
                # cycle if it's a HyperOpt type, should be caught by expression definition preproc
                # if input is list
                    # if list of lists
                        # assert that length of list is number of layers
                        # if expression type is list
                            # assert list is number of layers
                        # else
                            # expand expression type
                        # flag this parameter as having layer-wise HyperOpt
                        # loop entries of list
                            # if list
                                # treat sublist as HyperOpt expression definitions
                            # else
                                # treat single values as fixed, do error checking
                        # create list of flags denoting which layers have HyperOpt definitions
                        # add new entries to HyperOpt space with numbered index suffixes and HyperOpt definition
                    # else
                        # if expression type is present
                            # treat input as HyperOpt expression definition
                        # if no expression type is present
                            # treat input as fixed input, assert that length is number of layers

                # else
                    # if expression type is present
                        # throw error, even if "choice" is requested it's pointless and confusing
                    # if no expression type is present
                        # error checking
                        # expand input to number of layers
                
            # not running HyperOpt
            else:
                if isinstance(input_value, list):
                    if any(isinstance(elem, list) for elem in input_value):
                        raise ValueError("If not using HyperOpt, cannot use list of lists inputs for " + input_key)
                    assert len(input_value) == self.num_layers, (
                        "List input for " + input_key + " must have length " + str(self.num_layers))
                    input_values_assign = input_value
                else:
                    # expand input to number of layers
                    input_values_assign = [input_value] * self.num_layers

            # assign completed list
            param_space[input_key] = input_values_assign

    def preproc_layer_input(self, input_dict, params):
        """Do error checking and input expansions on layer inputs.
        
        If not using HyperOpt, then everything is already set by preproc_inputs and everything just gets moved into layer_params_list.
        If using HyperOpt, this expands single inputs and 
        """

        if input_dict["use_hyperopt"]:
            raise ValueError("HyperOpt preproc_layer_input not implemented yet")
        

        for layer_idx in range(self.num_layers):
            layer_dict = {}
            for input_key in self.input_keys:
                param_name = input_key[self.len_prefix + 1:]
                layer_dict[param_name] = params[input_key][layer_idx]
            self.layer_params_list.append(layer_dict)
                

    def assemble(self, input_dict, params, input_shape, batch_size=None):

        # set layer parameters, handling HyperOpt
        if not ((self.param_prefix == "decoder") and input_dict["mirrored_decoder"]):
            self.preproc_layer_input(input_dict, params)

        # start off with input layer
        self.layer_list.append(self.mllib.get_input_layer(input_shape, batch_size=batch_size, name="input_0"))
        self.num_layers += 1

        input_count = 1
        conv_count, trans_conv_count = 0, 0
        dense_count = 0
        reshape_count, flatten_count = 0, 0
        self.num_addtl_layers = 0
        addtl_layer_idxs = []
        for layer_idx, layer_dict in enumerate(self.layer_params_list):

            # TODO: check that this works with non-sequential topologies
            layer_type = layer_dict["layer_type"]

            layer_input_idx = layer_dict["layer_input_idx"]
            # handle situations where layer is silently added (e.g. flatten for dense layer)
            if layer_input_idx != -1:
                layer_input_idx -= self.addtl_layers
            layer_input = self.layer_list[layer_input_idx] 

            # TODO: add input layer capabilities

            # convolution layer
            if layer_type == "conv":
                dims = self.mllib.get_tensor_dims(layer_input) - 2  # ignore batch and channel dimensions
                layer_output = self.mllib.get_conv_layer(
                    layer_input,
                    dims,
                    layer_dict["num_filters"],
                    layer_dict["kern_size"],
                    layer_dict["strides"],
                    input_dict["network_order"],
                    activation=layer_dict["activation"],
                    padding=layer_dict["padding"],
                    kern_reg=layer_dict["kern_reg"],
                    kern_reg_val=layer_dict["kern_reg_val"],
                    act_reg=layer_dict["act_reg"],
                    act_reg_val=layer_dict["act_reg_val"],
                    bias_reg=layer_dict["bias_reg"],
                    bias_reg_val=layer_dict["bias_reg_val"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                    name="conv" + str(dims) + "d_" + str(conv_count),
                )
                conv_count += 1

            # transpose convolution layer
            elif layer_type == "trans_conv":
                dims = self.mllib.get_tensor_dims(layer_input) - 2  # ignore batch and channel dimensions
                assert dims > 0, "Input to transpose convolution must have rank greater than 2. Did you forget a reshape layer?"
                layer_output = self.mllib.get_trans_conv_layer(
                    layer_input,
                    dims,
                    layer_dict["num_filters"],
                    layer_dict["kern_size"],
                    layer_dict["strides"],
                    input_dict["network_order"],
                    activation=layer_dict["activation"],
                    padding=layer_dict["padding"],
                    kern_reg=layer_dict["kern_reg"],
                    kern_reg_val=layer_dict["kern_reg_val"],
                    act_reg=layer_dict["act_reg"],
                    act_reg_val=layer_dict["act_reg_val"],
                    bias_reg=layer_dict["bias_reg"],
                    bias_reg_val=layer_dict["bias_reg_val"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                    name="transconv" + str(dims) + "d_" + str(trans_conv_count),
                )
                trans_conv_count += 1

            # dense/fully-connected layer
            elif layer_type == "dense":
                layer_output, added_flatten = self.mllib.get_dense_layer(
                    layer_input,
                    layer_dict["output_size"],
                    activation=layer_dict["activation"],
                    kern_reg=layer_dict["kern_reg"],
                    kern_reg_val=layer_dict["kern_reg_val"],
                    act_reg=layer_dict["act_reg"],
                    act_reg_val=layer_dict["act_reg_val"],
                    bias_reg=layer_dict["bias_reg"],
                    bias_reg_val=layer_dict["bias_reg_val"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                    name="dense_" + str(dense_count),
                    flatten_count=flatten_count,
                )
                dense_count += 1
                if added_flatten:
                    addtl_layer_idxs.append(layer_idx + self.num_addtl_layers)
                    self.num_addtl_layers += 1
                    flatten_count += 1

            # reshape layer
            elif layer_type == "reshape":
                layer_output = self.mllib.get_reshape_layer(
                    layer_input,
                    layer_dict["target_shape"],
                    name="reshape_" + str(reshape_count),
                )
                reshape_count += 1

            # flatten layer
            elif layer_type == "flatten":
                layer_output = self.mllib.get_flatten_layer(
                    layer_input,
                    name="flatten_" + str(flatten_count),
                )
                flatten_count += 1

            else:
                raise ValueError("Invalid layer_type: " + str(layer_type))

            self.layer_list.append(layer_output)

        # account for any layers added silently
        self.num_layers += self.num_addtl_layers
        for layer_idx in addtl_layer_idxs:
            self.layer_params_list.insert(layer_idx, "addtl_layer")

        # finalize model object
        self.model_obj = self.mllib.build_model_obj(self.layer_list)        

