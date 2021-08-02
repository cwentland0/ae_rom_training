import os
from time import time
import pickle

from numpy import nan
from hyperopt import STATUS_OK

from ae_rom_training.constants import LAYER_PARAM_DICT, TRAIN_PARAM_DICT
from ae_rom_training.preproc_utils import remove_prefix
from ae_rom_training.hyperopt_utils import hp_expression


class AEROM:
    """Base class for autoencoder-based ROMs.

    Should always have an encoder and decoder, w/ optional time-stepper and/or parameter predictor.
    """

    def __init__(self, input_dict, mllib, network_suffix):

        self.model_dir = input_dict["model_dir"]
        self.latent_dim = input_dict["latent_dim"]
        self.train_separate = input_dict["train_separate"]

        self.mllib = mllib
        self.network_suffix = network_suffix
        self.hyperopt_param_names = []

        # have to instantiate this here as well because it's part of the build_and_train call
        self.param_space = {}

        if input_dict["train_ts"] and (self.time_stepper is None):
            raise ValueError("train_ts = True, but " + input_dict["aerom_type"] + " does not have a time stepper")

        # concatenate component networks
        self.component_networks = self.autoencoder.component_networks.copy()
        if self.time_stepper is not None:
            self.component_networks += self.time_stepper.component_networks.copy()

    def check_param_vs_layers(self, param, param_list_clear, layer_list_full, layer_list_clear):

        if (param in param_list_clear) and (not any(layer in layer_list_full for layer in layer_list_clear)):
            return True
        else:
            return False

    def preproc_inputs(self, input_dict):
        """Set up parameter space for layer and training inputs.

        For HyperOpt, preprocesses all inputs for component networks into HyperOpt expressions.
        """

        # clear out dict, since this may need to be reset for separate TS training
        self.param_space.clear()

        # TODO: This doesn't need to be repeated every time an autoencoder is instantiated
        # TODO: layer_input_idx shouldn't be included in Hyperopt space

        # layer parameters for each component network
        for network in self.component_networks:

            # don't assign a unique parameter space for a mirrored decoder
            if (network.param_prefix == "decoder") and input_dict["mirrored_decoder"]:
                continue

            network.input_keys = [key for key, _ in input_dict.items() if key.startswith(network.param_prefix + "_")]
            layer_types = input_dict[network.param_prefix + "_layer_type"]
            network.num_layers = len(layer_types)  # excluding input layer

            for param_name, param_list in LAYER_PARAM_DICT.items():

                input_key = network.param_prefix + "_" + param_name

                # parameter provided by user
                if input_key in input_dict:
                    input_value = input_dict[input_key]

                    if input_dict["use_hyperopt"]:

                        # HyperOpt types are handled by expression definition preproc
                        if input_key.endswith("_expr"):
                            continue
                        expr_key = input_key + "_expr"
                        if expr_key in network.input_keys:
                            expr_type = input_dict[expr_key]

                        # list input
                        if isinstance(input_value, list):

                            # list of lists input
                            if any([isinstance(elem, list) for elem in input_value]):

                                if all([isinstance(elem, list) for elem in input_value]):

                                    # list of Hyperopt expression definitions for each layer
                                    if isinstance(expr_type, list):
                                        assert len(expr_type) == network.num_layers, (
                                            "List input for "
                                            + expr_key
                                            + " must have length "
                                            + str(network.num_layers)
                                        )
                                        input_values_assign = [
                                            hp_expression(input_key + str(idx), expr_type[idx], val)
                                            for idx, val in enumerate(input_value)
                                        ]

                                    # list of fixed layer inputs to be chosen from
                                    else:
                                        assert all([(len(val) == network.num_layers) for val in input_value]), (
                                            "Sublists of list input for "
                                            + expr_key
                                            + " must have length "
                                            + str(network.num_layers)
                                        )
                                        input_values_assign = hp_expression(input_key, "choice", input_value)

                                    network.hyperopt_param_names.append(input_key)

                                else:
                                    raise ValueError("If list parameter contains lists, all elements must be lists.")

                            # single list input
                            # This defines either a HyperOpt expression or fixed layer inputs
                            else:

                                # HyperOpt expression definition
                                if expr_key in network.input_keys:
                                    input_values_assign = hp_expression(input_key, expr_type, input_value)
                                    network.hyperopt_param_names.append(input_key)

                                # fixed layer inputs
                                else:
                                    assert len(input_value) == network.num_layers, (
                                        "List input for " + input_key + " must have length " + str(network.num_layers)
                                    )
                                    input_values_assign = hp_expression(input_key, "choice", [input_value])

                        # single-value input
                        else:
                            if expr_key in network.input_keys:
                                # throw error, even if "choice" is requested it's pointless and confusing
                                # TODO: error message
                                raise ValueError
                            else:
                                # this will get expanded when building the network
                                input_values_assign = hp_expression(input_key, "choice", [input_value])

                    # not running HyperOpt
                    else:
                        if isinstance(input_value, list):
                            if any(isinstance(elem, list) for elem in input_value):
                                raise ValueError(
                                    "If not using HyperOpt, cannot use list of lists inputs for " + input_key
                                )

                        # list length checking happens later
                        input_values_assign = input_value

                # if not provided by user, try to assign default
                else:
                    default = param_list[1]

                    # ignore parameters without defaults if their necessitating layers aren't in the network
                    if (
                        self.check_param_vs_layers(
                            param_name, ["output_size"], layer_types, ["dense", "koopman_continous"]
                        ) or self.check_param_vs_layers(
                            param_name, ["num_filters", "kern_size", "dilation", "strides"], layer_types, ["conv", "trans_conv"]
                        ) or self.check_param_vs_layers(param_name, [], layer_types, ["conv", "trans_conv"])
                        or self.check_param_vs_layers(param_name, ["target_shape"], layer_types, ["reshape"])
                        or self.check_param_vs_layers(param_name, ["return_sequences"], layer_types, ["lstm", "tcn"])
                        or self.check_param_vs_layers(param_name, ["dilation_tcn"], layer_types, ["tcn"])
                    ):
                        continue

                    # no default exists and is required
                    elif default is nan:
                        raise ValueError(
                            "Input "
                            + input_key
                            + " has no default and must be provided. "
                            + "If your model doesn't need it, just provide a dummy value for now."
                        )

                    # a default exists
                    else:
                        if input_dict["use_hyperopt"]:
                            input_values_assign = hp_expression(input_key, "choice", [default])
                        else:
                            input_values_assign = default

                        # add parameter to input keys
                        network.input_keys.append(input_key)

                # assign completed list
                self.param_space[input_key] = input_values_assign

        # training parameters
        # NOTE: train_prefix is NOT included in self.param_space keys
        for param_name, param_list in TRAIN_PARAM_DICT.items():

            default = param_list[1]

            param_key = self.train_prefix + param_name
            expr_key = param_key + "_expr"
            expr_type = None
            if expr_key in input_dict:
                expr_type = input_dict[expr_key]

            if self.time_stepper is None:
                if param_name in ["seq_length"]:
                    continue

            # has no default
            if default is nan:

                # es_patience has no default, but is required if early_stopping = True
                if (param_name == "es_patience") and ("early_stopping" not in self.param_space):
                    continue

                try:
                    param_val = input_dict[param_key]
                except KeyError:
                    raise KeyError(param_key + " has no default value, you must provide a value")

            else:

                try:
                    param_val = input_dict[param_key]
                except KeyError:
                    param_val = default

            if input_dict["use_hyperopt"]:

                # If providing an expression type, input must be a list
                if expr_type is not None:
                    assert isinstance(param_val, list), (
                        "When using HyperOpt and providing "
                        + expr_key
                        + " for training input "
                        + param_key
                        + ", input must be a list"
                    )

                    self.param_space[param_name] = hp_expression(param_key, expr_type, param_val)
                    self.hyperopt_param_names.append(param_key)

                # If not providing an expression, input cannot be a list
                else:

                    assert not isinstance(param_val, list), (
                        "When using HyperOpt and not providing "
                        + expr_key
                        + " for training input "
                        + param_key
                        + " input cannot be a list"
                    )

                    self.param_space[param_name] = hp_expression(param_key, "choice", [param_val])

            # none of these should be list inputs when not using HyperOpt
            else:
                assert not isinstance(param_val, list), (
                    "When not using HyperOpt, training input " + param_key + " cannot be a list"
                )

                self.param_space[param_name] = param_val

    def build_and_train(self, params, input_dict, data_list_train, data_list_val, training_ae=False, training_ts=False):
        """Build and train network.

        Acts as objective function for HyperOpt, or normal training function without HyperOpt.
        """

        # keep track of what we're training on this run
        self.training_ae = training_ae
        self.training_ts = training_ts

        # just here in case I make a calling mistake in the future
        assert self.training_ae or self.training_ts, "Must train autoencoder, time-stepper, or both."
        
        # prefix for grabbing training parameters
        if self.training_ae:
            if self.training_ts:
                self.train_prefix = ""
            else:
                self.train_prefix = "ae_"
        else:
            self.train_prefix = "ts_"

        # aggregate 
        self.preproc_inputs(input_dict)

        # Keep track of Hyperopt trial number
        if input_dict["use_hyperopt"]:
            print(
                "\nHYPEROPT TRIAL NUMBER "
                + str(input_dict["trial_number"])
                + "/"
                + str(input_dict["hyperopt_max_evals"])
                + "\n"
            )
            input_dict["trial_number"] += 1

        # build autoencoder
        # always have an autoencoder, even if it's just loaded from disk
        assert self.autoencoder is not None, "Autoencoder not initialized for this model"
        data_shape = data_list_train[0].shape[1:]
        self.autoencoder.build(
            input_dict, params, data_shape, self.training_ae, self.training_ts, self.network_suffix, batch_size=None
        )
        self.autoencoder.check_build(input_dict, params, data_shape)

        # build time-stepper, if requested
        if self.training_ts:
            self.time_stepper.build(input_dict, params, batch_size=None)
            self.time_stepper.check_build(input_dict, params)

        self.build()  # finish building

        if input_dict["use_hyperopt"]:
            self.print_aerom_hyperopt_sample(params)

        # train network, finally
        time_start = time()
        loss_train, loss_val = self.train(input_dict, params, data_list_train, data_list_val)
        train_time = time() - time_start
        print("=================================================================")
        print("Training complete in " + str(train_time) + " seconds")
        print("=================================================================")

        # check if this model is the best so far, if so save
        self.check_best(input_dict, loss_val, params)

        # return optimization info dictionary
        return {
            "loss": loss_train,  # training loss at end of training
            "true_loss": loss_val,  # validation loss at end of training
            "status": STATUS_OK,  # check for correct exit
            "eval_time": train_time,  # time (in seconds) to train model
        }

    def train(self, input_dict, params, data_list_train, data_list_val):
        """Train the network.

        If the entire model, along with its training scheme, can be lumped into a single model,
        then train_model_builtin should be used.
        If the model requires customized training or can't be evaluated as a single model object,
        then train_model_custom should be used.
        train_builtin should be set in child class __init__ accordingly
        """

        # get training objects
        loss = self.mllib.get_loss_function(params)
        optimizer = self.mllib.get_optimizer(params)
        options = self.mllib.get_options(params)

        # Built-in training method implemented in ML library
        if self.training_ae != self.training_ts:
            loss_train, loss_val = self.train_model_builtin(
                data_list_train, data_list_val, optimizer, loss, options, input_dict, params
            )

        # Custom training method is implemented by child class
        else:
            loss_train, loss_val = self.train_model_custom(
                data_list_train, data_list_val, optimizer, loss, options, input_dict, params
            )

        return loss_train, loss_val

    def print_aerom_hyperopt_sample(self, params):
        """Display current Hyperopt selection"""

        print("\n=================================================================")
        print("CURRENT PARAMETER SAMPLE")
        print("=================================================================")

        if self.hyperopt_param_names:
            print("-------------------")
            print("training parameters")
            print("-------------------")
            for param_key in self.hyperopt_param_names:
                print("-> " + param_key + ": " + str(params[param_key]))

        if self.training_ae:
            self.print_component_hyperopt_sample(params, self.autoencoder)
        if self.training_ts:
            self.print_component_hyperopt_sample(params, self.time_stepper)

        print("=================================================================\n")

    def print_component_hyperopt_sample(self, params, component):

        for network in component.component_networks:
            if network.hyperopt_param_names:
                print("-------------------")
                print(network.param_prefix + " parameters")
                print("-------------------")
                for param_key in network.hyperopt_param_names:
                    print("-> " + param_key + ": " + str(params[param_key]))

    def check_best(self, input_dict, loss_val, params):

        model_dir = input_dict["model_dir"]
        loss_loc = os.path.join(model_dir, self.train_prefix + "val_loss" + self.network_suffix + ".dat")

        # open minimum loss file
        try:
            with open(loss_loc) as f:
                min_loss = float(f.read().strip())
        except FileNotFoundError:
            print("First iteration, writing best model")
            min_loss = loss_val + 1.0  # first time, guarantee that loss is written to file

        # if current model validation loss beats previous best, overwrite
        if loss_val < min_loss:
            print("New best found! Overwriting...")

            self.save(model_dir)

            params_loc = os.path.join(model_dir, self.train_prefix + "params" + self.network_suffix + ".pickle")
            with open(params_loc, "wb") as f:
                pickle.dump(params, f)

            with open(loss_loc, "w") as f:
                f.write(str(loss_val) + "\n")
