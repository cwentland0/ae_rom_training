import os
from time import time
import pickle

from numpy import nan
# from hyperopt import STATUS_OK
STATUS_OK = True

from ae_rom_training.constants import TRAIN_PARAM_NAMES, TRAIN_PARAM_DEFAULTS, TRAIN_PARAM_DTYPES
from ae_rom_training.preproc_utils import catch_input
from ae_rom_training.ml_model.encoder import Encoder
from ae_rom_training.ml_model.decoder import Decoder

class Autoencoder():
    """Base class for autoencoders.
    
    Does not inherit from MLModel since it's an assembly of models.
    Always has an encoder and decoder, w/ optional time-stepper or parameter predictor.
    """

    def __init__(self, input_dict, mllib, network_suffix):

        self.model_dir = input_dict["model_dir"]
        self.mllib = mllib
        self.network_suffix = network_suffix
        self.param_space = {}

        self.preproc_training_inputs(input_dict)

    def preproc_training_inputs(self, input_dict):
        """Set up parameter space for training inputs."""

        # TODO: This doesn't need to be repeated every time an autoencoder is instantiated

        if input_dict["use_hyperopt"]:
            raise ValueError("Training input HyperOpt not implemented yet")

        else:

            for param_idx, param_name in enumerate(TRAIN_PARAM_NAMES):
            
                default = TRAIN_PARAM_DEFAULTS[param_idx]
                if default is nan:
                    # es_patience has no default, but is required if early_stopping = True
                    if (param_name == "es_patience") and not self.param_space["early_stopping"]:
                        continue
                    
                    self.param_space[param_name] = input_dict[param_name]

                else:
                    self.param_space[param_name] = catch_input(input_dict, param_name, default)

    def build_and_train(self, params, input_dict, data_train, data_val):
        """Build and train full autoencoder.
        
        Acts as objective function for HyperOpt, or normal training function without HyperOpt.
        """

        # build network
        data_shape = data_train.shape[1:]
        self.model = self.build(input_dict, params, data_shape, batch_size=None)  # must be implicit batch for training
        self.check_build(input_dict, data_shape)

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