from ae_rom_training.constants import LAYER_PARAM_DICT

# TODO: handle different network definitions for separate networks


class MLModel:
    """Base class for component ML models that make up the autoencoder.

    Examples include the encoder and decoder, as well as time steppers and parameter predictors.

    Args:
        input_dict: dict of input values from input text file.
        param_space: parameter space of containing Autoencoder object, which may contain HyperOpt expressions
        mllib: MLLibrary object defining machine learning library functionalities
    """

    def __init__(self, net_idx, param_prefix, mllib):

        self.net_idx = net_idx
        self.param_prefix = param_prefix
        self.mllib = mllib
        self.len_prefix = len(self.param_prefix)
        self.hyperopt_param_names = []

    def preproc_layer_input(self, input_dict, params):
        """Do error checking and input expansions on layer inputs.

        If not using HyperOpt, then everything is already set by preproc_inputs
        and everything just gets moved into layer_params_list.
        """

        self.layer_params_list = []

        # Get layer lists or expand single inputs
        # params has already been preprocessed to be in the proper format
        for input_key in self.input_keys:

            if input_key.endswith("_expr"):
                continue

            param_type = input_key[self.len_prefix + 1 :]
            dtype = LAYER_PARAM_DICT[param_type][0]

            input_value = params[input_key]

            # Check that length of list inputs matches number of layers
            # Lists only appear when explicitly specified when NOT using Hyperopt
            if isinstance(input_value, list):
                assert len(input_value) == self.num_layers, (
                    "List input for " + input_key + " must have length " + str(self.num_layers)
                )

            # Hyperopt converts all iterable parameters to tuples,
            # so need to do some corrections and error checking here
            elif isinstance(input_value, tuple) and input_dict["use_hyperopt"]:

                # the input is expected to be a tuple
                if isinstance(dtype, tuple):
                    if input_key in self.hyperopt_param_names:
                        assert not (len(input_value) == self.num_layers), "Broken edge case"
                        params[input_key] = [input_value] * self.num_layers
                    else:
                        assert len(input_value) == self.num_layers, (
                            "List input for " + input_key + " must have length " + str(self.num_layers)
                        )
                        params[input_key] = list(input_value)

                # this was input as a list and Hyperopt converted it to a tuple, convert back
                else:
                    if isinstance(dtype, list):
                        assert not (len(input_value) == self.num_layers), "Broken edge case"
                        params[input_key] = [list(input_value)] * self.num_layers
                    else:
                        assert len(input_value) == self.num_layers, (
                            "List input for " + input_key + " must have length " + str(self.num_layers)
                        )
                        params[input_key] = list(input_value)

            # expand input to number of layers
            else:
                params[input_key] = [input_value] * self.num_layers

            # if -1 is entered for input size, assume this is supposed to be equal to latent_dim
            # makes training automation a lot easier
            if param_type == "output_size":
                params[input_key] = [
                    input_dict["latent_dim"][self.net_idx] if x == -1 else x for x in params[input_key]
                ]

        for layer_idx in range(self.num_layers):
            layer_dict = {}
            for input_key in self.input_keys:
                if input_key.endswith("_expr"):
                    continue
                param_name = input_key[self.len_prefix + 1 :]
                layer_dict[param_name] = params[input_key][layer_idx]
            self.layer_params_list.append(layer_dict)

    def assemble(self, input_dict, params, input_shape, batch_size=None):

        # set layer parameters, handling HyperOpt
        if not ((self.param_prefix == "decoder") and input_dict["mirrored_decoder"]):
            self.preproc_layer_input(input_dict, params)

        # start off with input layer
        input_idx_list = [0]
        self.layer_list = []
        self.layer_list.append(self.mllib.get_input_layer(input_shape, batch_size=batch_size, name="input_0"))
        self.num_layers_total = self.num_layers
        self.num_layers_total += 1

        input_count = 1
        conv_count, trans_conv_count = 0, 0
        dense_count = 0
        koopman_continuous_count = 0
        lstm_count, gru_count, tcn_count = 0, 0, 0
        reshape_count, flatten_count = 0, 0
        self.num_addtl_layers = 0
        addtl_layer_idxs = []
        for layer_idx, layer_dict in enumerate(self.layer_params_list):

            # TODO: check that this works with non-sequential topologies
            layer_type = layer_dict["layer_type"]

            layer_input_idx = layer_dict["layer_input_idx"]
            # handle situations where layer is silently added (e.g. flatten for dense layer)
            if layer_input_idx != -1:
                layer_input_idx -= self.num_addtl_layers
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
                    layer_dict["dilation"],
                    input_dict["network_order"],
                    layer_dict["activation"],
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
                assert (
                    dims > 0
                ), "Input to transpose convolution must have rank greater than 2. Did you forget a reshape layer?"
                layer_output = self.mllib.get_trans_conv_layer(
                    layer_input,
                    dims,
                    layer_dict["num_filters"],
                    layer_dict["kern_size"],
                    layer_dict["strides"],
                    layer_dict["dilation"],
                    input_dict["network_order"],
                    layer_dict["activation"],
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
                    layer_dict["activation"],
                    use_bias=layer_dict["use_bias"],
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

            # custom continuous Koopman operator layer
            # requires and extra input layer for input time step
            elif layer_type == "koopman_continuous":
                layer_output = self.mllib.get_continuous_koopman_layer(
                    layer_input,
                    layer_dict["output_size"],
                    input_count,
                    stable=layer_dict["stable"],
                    kern_init=layer_dict["kern_init"],
                    name="koopman_continuous_" + str(koopman_continuous_count),
                )
                addtl_layer_idxs.append(layer_idx + self.num_addtl_layers)
                input_idx_list.append(layer_idx + self.num_addtl_layers + 1)
                self.num_addtl_layers += 1
                input_count += 1
                koopman_continuous_count += 1

            elif layer_type == "lstm":
                layer_output = self.mllib.get_lstm_layer(
                    layer_input,
                    layer_dict["output_size"],
                    layer_dict["return_sequences"],
                    layer_dict["activation"],
                    recurrent_activation=layer_dict["recurrent_activation"],
                    use_bias=layer_dict["use_bias"],
                    kern_reg=layer_dict["kern_reg"],
                    kern_reg_val=layer_dict["kern_reg_val"],
                    act_reg=layer_dict["act_reg"],
                    act_reg_val=layer_dict["act_reg_val"],
                    bias_reg=layer_dict["bias_reg"],
                    bias_reg_val=layer_dict["bias_reg_val"],
                    recurrent_reg=layer_dict["recurrent_reg"],
                    recurrent_reg_val=layer_dict["recurrent_reg_val"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                    recurrent_init=layer_dict["recurrent_init"],
                    name="lstm_" + str(lstm_count),
                )
                lstm_count += 1

            elif layer_type == "gru":
                layer_output = self.mllib.get_gru_layer(
                    layer_input,
                    layer_dict["output_size"],
                    layer_dict["return_sequences"],
                    layer_dict["activation"],
                    recurrent_activation=layer_dict["recurrent_activation"],
                    use_bias=layer_dict["use_bias"],
                    kern_reg=layer_dict["kern_reg"],
                    kern_reg_val=layer_dict["kern_reg_val"],
                    act_reg=layer_dict["act_reg"],
                    act_reg_val=layer_dict["act_reg_val"],
                    bias_reg=layer_dict["bias_reg"],
                    bias_reg_val=layer_dict["bias_reg_val"],
                    recurrent_reg=layer_dict["recurrent_reg"],
                    recurrent_reg_val=layer_dict["recurrent_reg_val"],
                    kern_init=layer_dict["kern_init"],
                    bias_init=layer_dict["bias_init"],
                    recurrent_init=layer_dict["recurrent_init"],
                    name="gru_" + str(gru_count),
                )
                gru_count += 1

            elif layer_type == "tcn":
                layer_output = self.mllib.get_tcn_layer(
                    layer_input,
                    layer_dict["num_filters"],
                    layer_dict["kern_size"],
                    layer_dict["activation"],
                    layer_dict["dilation_tcn"],
                    layer_dict["return_sequences"],
                    padding=layer_dict["padding_tcn"],
                    kern_init=layer_dict["kern_init"],
                    name="tcn_" + str(tcn_count),
                )
                tcn_count += 1

            # reshape layer
            elif layer_type == "reshape":
                layer_output = self.mllib.get_reshape_layer(
                    layer_input, layer_dict["target_shape"], name="reshape_" + str(reshape_count),
                )
                reshape_count += 1

            # flatten layer
            elif layer_type == "flatten":
                layer_output = self.mllib.get_flatten_layer(layer_input, name="flatten_" + str(flatten_count),)
                flatten_count += 1

            else:
                raise ValueError("Invalid layer_type: " + str(layer_type))

            if isinstance(layer_output, list):
                self.layer_list += layer_output
            else:
                self.layer_list.append(layer_output)

        # account for any layers added silently
        self.num_layers_total += self.num_addtl_layers
        for layer_idx in addtl_layer_idxs:
            self.layer_params_list.insert(layer_idx, "addtl_layer")

        # finalize model object
        # TODO: handle multiple outputs
        self.model_obj = self.mllib.build_model_obj(self.layer_list, input_idx_list)
