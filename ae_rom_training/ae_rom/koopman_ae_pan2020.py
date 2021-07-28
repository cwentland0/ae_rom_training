import os

import numpy as np

from ae_rom_training.preproc_utils import hankelize, calc_time_values
from ae_rom_training.ae_rom.ae_rom import AEROM
from ae_rom_training.autoencoder.autoencoder import Autoencoder
from ae_rom_training.time_stepper.koopman import Koopman
from ae_rom_training.constants import FLOAT_TYPE


class KoopmanAEPan2020(AEROM):
    """Autoencoder which learns discrete Koopman, via Otto and Rowley (2019)"""

    def __init__(self, input_dict, mllib, network_suffix):

        self.autoencoder = Autoencoder(mllib)
        self.time_stepper = Koopman(input_dict, mllib, continuous=True)

        # check that time inputs are present
        assert input_dict["time_init_list_train"][0] is not None, "Must provide time_init_list_train for continuous Koopman"
        assert input_dict["dt_list_train"][0] is not None, "Must provide dt_list_train for continuous Koopman"
        if input_dict["data_files_val"][0] is not None:
            assert input_dict["time_init_list_val"][0] is not None, "Must provide time_init_list_val for continuous Koopman"
            assert input_dict["dt_list_val"][0] is not None, "Must provide dt_list_val for continuous Koopman"

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
        data_list_train,
        data_list_val,
        optimizer,
        loss,
        options,
        input_dict,
        params,
        param_prefix,
    ):
        """Call custom training loop after organizing data"""

        # get solution time series windows
        seq_length = params["seq_length"]
        seq_step = params["seq_step"]
        data_list_train_seqs = hankelize(data_list_train, seq_length, seq_step=seq_step)
        data_list_val_seqs = hankelize(data_list_val, seq_length, seq_step=seq_step)

        # get time value windows
        time_values_list_train, time_diffs_list_train = [], []
        time_values_list_val, time_diffs_list_val = [], []
        # if split came directly from training sets
        if input_dict["data_files_val"][0] is None:
            for set_idx in range(len(data_list_train)):
                time_values, _ = calc_time_values(
                    input_dict["time_init_list_train"][set_idx],
                    input_dict["dt_list_train"][set_idx],
                    input_dict["idx_start_list_train"][set_idx],
                    input_dict["idx_end_list_train"][set_idx],
                    input_dict["idx_skip_list_train"][set_idx],
                )
                time_values_list_train.append(time_values[input_dict["split_idxs_train"][set_idx]])
                time_values_list_val.append(time_values[input_dict["split_idxs_val"][set_idx]])
        else:
            raise ValueError("Time value calcs for separate val sets not implemented")

        time_values_list_train_seqs = hankelize(time_values_list_train, seq_length, seq_step=seq_step)
        time_values_list_val_seqs = hankelize(time_values_list_val, seq_length, seq_step=seq_step)

        # concatenate and shuffle sequences
        data_train_seqs = np.concatenate(data_list_train_seqs, axis=0)
        data_val_seqs = np.concatenate(data_list_val_seqs, axis=0)
        time_values_train_seqs = np.concatenate(time_values_list_train_seqs, axis=0)
        time_values_val_seqs = np.concatenate(time_values_list_val_seqs, axis=0)
        shuffle_idxs = np.random.permutation(np.arange(data_train_seqs.shape[0]))
        data_train_seqs = data_train_seqs[shuffle_idxs, ...]
        time_values_train_seqs = time_values_train_seqs[shuffle_idxs, ...]

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
            continuous=True,
            time_values_train=time_values_train_seqs,
            time_values_val=time_values_val_seqs,
        )

        self.loss_train_recon = loss_addtl_train_list[0]
        self.loss_train_step = loss_addtl_train_list[1]
        self.loss_val_recon = loss_addtl_val_list[0]
        self.loss_val_step = loss_addtl_val_list[1]

        loss_train = loss_train_hist[-1]
        loss_val = loss_val_hist[-1]

        return loss_train, loss_val
