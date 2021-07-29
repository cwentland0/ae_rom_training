import os

import numpy as np

from ae_rom_training.preproc_utils import hankelize
from ae_rom_training.ae_rom.ae_rom import AEROM
from ae_rom_training.autoencoder.autoencoder import Autoencoder
from ae_rom_training.time_stepper.koopman import Koopman


class KoopmanAEOtto2019(AEROM):
    """Autoencoder which learns discrete Koopman, via Otto and Rowley (2019)"""

    def __init__(self, input_dict, mllib, network_suffix):

        self.autoencoder = Autoencoder(mllib)
        self.time_stepper = Koopman(input_dict, mllib, continuous=False)

        super().__init__(input_dict, mllib, network_suffix)

        # TODO: handle separate training, blech
        self.train_builtin = False

    def build(self):
        """Assemble singular model object for entire network, if possible"""

        # TODO: need to figure out how to handle separate training while retaining this class object
        #   In the case of separate training, would want to compile the autoencoder
        pass

    def save(self, model_dir):
        """Save individual networks in AE ROM"""

        # TODO: again, should be able to handle selective saving for separate training
        self.autoencoder.save(model_dir, self.network_suffix)
        self.time_stepper.save(model_dir, self.network_suffix)

        loss_train_recon_path = os.path.join(model_dir, "loss_train_recon" + self.network_suffix + ".npy")
        loss_train_step_path = os.path.join(model_dir, "loss_train_step" + self.network_suffix + ".npy")
        loss_val_recon_path = os.path.join(model_dir, "loss_val_recon" + self.network_suffix + ".npy")
        loss_val_step_path = os.path.join(model_dir, "loss_val_step" + self.network_suffix + ".npy")
        np.save(loss_train_recon_path, self.loss_train_recon)
        np.save(loss_train_step_path, self.loss_train_step)
        np.save(loss_val_recon_path, self.loss_val_recon)
        np.save(loss_val_step_path, self.loss_val_step)

    def train_model_custom(
        self, data_list_train, data_list_val, optimizer, loss, options, params, input_dict, param_prefix,
    ):
        """Call custom training loop after organizing data"""

        # get time series windows
        seq_length = params["seq_length"]
        seq_step = params["seq_step"]
        data_list_train_seqs = hankelize(data_list_train, seq_length, seq_step=seq_step)
        data_list_val_seqs = hankelize(data_list_val, seq_length, seq_step=seq_step)

        # concatenate and shuffle sequences
        data_train_seqs = np.concatenate(data_list_train_seqs, axis=0)
        data_val_seqs = np.concatenate(data_list_val_seqs, axis=0)
        np.random.shuffle(data_train_seqs)

        # get source list
        self.grad_source_list = [
            self.autoencoder.decoder.model_obj,
            self.autoencoder.encoder.model_obj,
            self.time_stepper.stepper.model_obj,
        ]

        loss_train_hist, loss_val_hist, loss_addtl_train_list, loss_addtl_val_list = self.mllib.train_model_custom(
            self,
            data_train_seqs,
            data_train_seqs,
            data_val_seqs,
            data_val_seqs,
            optimizer,
            loss,
            options,
            params,
            param_prefix,
        )

        self.loss_train_recon = loss_addtl_train_list[0]
        self.loss_train_step = loss_addtl_train_list[1]
        self.loss_val_recon = loss_addtl_val_list[0]
        self.loss_val_step = loss_addtl_val_list[1]

        # TODO: adjust this value if
        loss_train = loss_train_hist[-1]
        loss_val = loss_val_hist[-1]

        return loss_train, loss_val
