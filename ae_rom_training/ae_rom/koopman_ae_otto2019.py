import os

import numpy as np

from ae_rom_training.ae_rom.ae_rom import AEROM
from ae_rom_training.autoencoder.autoencoder import Autoencoder
from ae_rom_training.time_stepper.koopman_discrete import KoopmanDiscreteTS
from ae_rom_training.constants import FLOAT_TYPE


class KoopmanAEOtto2019(AEROM):
    """Autoencoder which learns discrete Koopman, via Otto and Rowley (2019)"""

    def __init__(self, input_dict, mllib, network_suffix):

        self.autoencoder = Autoencoder(mllib)
        self.time_stepper = KoopmanDiscreteTS(input_dict, mllib)

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
        self,
        data_train_input,
        data_train_output,
        data_val_input,
        data_val_output,
        optimizer,
        loss,
        options,
        params,
        param_prefix,
    ):
        """Call custom training loop after organizing data"""

        # TODO: put Hankelization and shuffle in its own function in preproc_utils

        seq_length = params["seq_length"]
        num_seqs_train = data_train_input.shape[0] - seq_length + 1
        num_seqs_val = data_val_input.shape[0] - seq_length + 1

        # get time series windows
        data_train_input_seqs = np.zeros((num_seqs_train, seq_length,) + data_train_input.shape[1:], dtype=FLOAT_TYPE)
        data_val_input_seqs = np.zeros((num_seqs_val, seq_length,) + data_val_input.shape[1:], dtype=FLOAT_TYPE)
        for seq_idx in range(num_seqs_train):
            data_train_input_seqs[seq_idx, ...] = data_train_input[seq_idx : seq_idx + seq_length, ...]
        for seq_idx in range(num_seqs_val):
            data_val_input_seqs[seq_idx, ...] = data_val_input[seq_idx : seq_idx + seq_length, ...]

        # shuffle sequences
        shuffle_idxs = np.random.permutation(num_seqs_train)
        data_train_input_seqs = data_train_input_seqs[shuffle_idxs, ...]

        # get source list
        self.grad_source_list = [
            self.autoencoder.decoder.model_obj,
            self.autoencoder.encoder.model_obj,
            self.time_stepper.stepper.model_obj,
        ]

        loss_train_hist, loss_val_hist, loss_addtl_train_list, loss_addtl_val_list = self.mllib.train_model_custom(
            self,
            data_train_input_seqs,
            data_train_input_seqs,
            data_val_input_seqs,
            data_val_input_seqs,
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
