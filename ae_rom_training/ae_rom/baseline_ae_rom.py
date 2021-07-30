import os
import numpy as np

from ae_rom_training.ae_rom.ae_rom import AEROM
from ae_rom_training.autoencoder.baseline_ae import BaselineAE


class BaselineAEROM(AEROM):
    """Simple AE ROM with only autoencoder (no time-stepper)"""

    def __init__(self, input_dict, mllib, network_suffix):

        self.autoencoder = BaselineAE(mllib)
        self.time_stepper = None

        super().__init__(input_dict, mllib, network_suffix)

    def build(self):
        """Assemble singular model object for entire network, if possible"""

        self.model_obj = self.autoencoder.model_obj

    def train_model_builtin(
        self, data_list_train, data_list_val, optimizer, loss, options, input_dict, params,
    ):

        # concatenate and shuffle data sets, since order doesn't matter
        data_train = np.concatenate(data_list_train, axis=0)
        data_val = np.concatenate(data_list_val, axis=0)
        np.random.shuffle(data_train)

        # train
        self.loss_train_hist, self.loss_val_hist, loss_train, loss_val = self.mllib.train_model_builtin(
            self.model_obj,
            data_train,
            data_train,
            data_val,
            data_val,
            optimizer,
            loss,
            options,
            input_dict,
            params,
            self.train_prefix,
        )

        return loss_train, loss_val

    def train_model_custom(self):

        raise ValueError("Custom training loop not implemented for BaselineAEROM")

    def save(self, model_dir):
        """Save individual networks in AE ROM"""

        self.autoencoder.save(model_dir, self.network_suffix)

        loss_train_hist_path = os.path.join(model_dir, "loss_train_hist" + self.network_suffix + ".npy")
        loss_val_hist_path = os.path.join(model_dir, "loss_val_hist" + self.network_suffix + ".npy")
        np.save(loss_train_hist_path, self.loss_train_hist)
        np.save(loss_val_hist_path, self.loss_val_hist)
