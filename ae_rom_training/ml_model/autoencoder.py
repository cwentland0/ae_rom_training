from time import time

from hyperopt import STATUS_OK

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

        # baseline encoder and decoder
        self.encoder = Encoder(input_dict, mllib)
        breakpoint()
        # self.decoder = Decoder(input_dict, mllib)

        # time-stepper, if requested

    def build_and_train(self, params, data_train, data_val):
        """Build and train full autoencoder.
        
        Acts as objective function for HyperOpt.
        """

        # build network
        self.model = self.build(params)  # must be implicit batch for training

        # train network
        time_start = time()
        loss_train, loss_val = self.train(params, data_train, data_val)
        eval_time = time() - time_start

        # check if this model is the best so far, if so save
        self.check_best()

        # return optimization info dictionary
        return {
            "loss": loss_train,  # training loss at end of training
            "true_loss": loss_val,  # validation loss at end of training
            "status": STATUS_OK,  # check for correct exit
            "eval_time": eval_time,  # time (in seconds) to train model
        }

    def build(self, params):

        # assemble encoder

        # assemble decoder (mirror, if requested)

        # assemble time stepper (if requested)

        pass

    def train(self, param_space):
        # print summary before training
        loss_train, loss_val = 1e-5, 1e-5
        return loss_train, loss_val

    def check_best(self):
        pass