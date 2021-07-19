import os
from time import time
import pickle

from numpy import nan
from hyperopt import STATUS_OK

from ae_rom_training.constants import TRAIN_PARAM_DICT
from ae_rom_training.hyperopt_utils import hp_expression


class Autoencoder:
    """Base class for autoencoders.

    Should always have an encoder and decoder, w/ optional time-stepper and/or parameter predictor.
    """

    def __init__(self, input_dict, mllib, network_suffix):

        self.model_dir = input_dict["model_dir"]
        self.mllib = mllib
        self.network_suffix = network_suffix
        self.param_space = {}
        self.hyperopt_param_names = []

        self.preproc_inputs(input_dict)

    def preproc_inputs(self, input_dict):
        """Set up parameter space for training inputs.

        For HyperOpt, preprocesses all inputs for component networks into HyperOpt expressions.
        """

        # TODO: This doesn't need to be repeated every time an autoencoder is instantiated
        # TODO: layer_input_idx shouldn't be included in Hyperopt space

        # layer parameters for each component network
        for network in self.component_networks:

            # don't assign a unique parameter space for a mirrored decoder
            if (network.param_prefix == "decoder") and input_dict["mirrored_decoder"]:
                continue

            network.input_keys = [key for key, _ in input_dict.items() if key.startswith(network.param_prefix + "_")]
            network.num_layers = len(input_dict[network.param_prefix + "_layer_type"])  # excluding input layer

            # network layer inputs
            for input_key in network.input_keys:
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
                                        "List input for " + expr_key + " must have length " + str(network.num_layers)
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
                            raise ValueError("If not using HyperOpt, cannot use list of lists inputs for " + input_key)

                    # list length checking happens later
                    input_values_assign = input_value

                # assign completed list
                self.param_space[input_key] = input_values_assign

        # training parameters
        for param_name, param_list in TRAIN_PARAM_DICT.items():

            default = param_list[1]
            expr_key = param_name + "_expr"
            expr_type = None
            if expr_key in input_dict:
                expr_type = input_dict[expr_key]

            # has no default
            if default is nan:

                # es_patience has no default, but is required if early_stopping = True
                if (param_name == "es_patience") and ("early_stopping" not in self.param_space):
                    continue

                try:
                    param_val = input_dict[param_name]
                except KeyError:
                    raise KeyError(param_name + " has no default value, you must provide a value")

            else:

                try:
                    param_val = input_dict[param_name]
                except KeyError:
                    param_val = default

            if input_dict["use_hyperopt"]:

                # If providing an expression type, input must be a list
                if expr_type is not None:
                    assert isinstance(param_val, list), (
                        "When using HyperOpt and providing "
                        + expr_key
                        + " for training input "
                        + param_name
                        + ", input must be a list"
                    )

                    self.param_space[param_name] = hp_expression(param_name, expr_type, param_val)
                    self.hyperopt_param_names.append(param_name)

                # If not providing an expression, input cannot be a list
                else:

                    assert not isinstance(param_val, list), (
                        "When using HyperOpt and not providing "
                        + expr_key
                        + " for training input "
                        + param_name
                        + " input cannot be a list"
                    )

                    self.param_space[param_name] = hp_expression(param_name, "choice", [param_val])

            # none of these should be list inputs when not using HyperOpt
            else:
                assert not isinstance(param_val, list), (
                    "When not using HyperOpt, training input " + param_name + " cannot be a list"
                )

                self.param_space[param_name] = param_val

    def build_and_train(self, params, input_dict, data_train, data_val):
        """Build and train full autoencoder.

        Acts as objective function for HyperOpt, or normal training function without HyperOpt.
        """

        # build network
        data_shape = data_train.shape[1:]
        self.model = self.build(input_dict, params, data_shape, batch_size=None)  # must be implicit batch for training
        self.check_build(input_dict, data_shape)

        # Display current Hyperopt selection
        if input_dict["use_hyperopt"]:
            print("========================")
            print("CURRENT PARAMETER SAMPLE")
            print("========================")
            print("training parameters")
            print("-------------------")
            for param_key in self.hyperopt_param_names:
                print(param_key + ": " + str(params[param_key]))
            for network in self.component_networks:
                print(network.param_prefix + " parameters")
                print("-------------------")
                for param_key in network.hyperopt_param_names:
                    print(param_key + ": " + str(params[param_key]))
            print("========================")

        # train network
        time_start = time()
        loss_train, loss_val = self.train(input_dict, params, data_train, data_val)
        eval_time = time() - time_start

        # check if this model is the best so far, if so save
        self.check_best(input_dict, loss_val, params)

        # return optimization info dictionary
        return {
            "loss": loss_train,  # training loss at end of training
            "true_loss": loss_val,  # validation loss at end of training
            "status": STATUS_OK,  # check for correct exit
            "eval_time": eval_time,  # time (in seconds) to train model
        }

    def check_best(self, input_dict, loss_val, params):

        model_dir = input_dict["model_dir"]
        loss_loc = os.path.join(model_dir, "val_loss" + self.network_suffix + ".dat")

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

            params_loc = os.path.join(model_dir, "params" + self.network_suffix + ".pickle")
            with open(params_loc, "wb") as f:
                pickle.dump(params, f)

            with open(loss_loc, "w") as f:
                f.write(str(loss_val) + "\n")
