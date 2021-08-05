import os

import numpy as np

from ae_rom_training.preproc_utils import window, hankelize
from ae_rom_training.ae_rom.ae_rom import AEROM
from ae_rom_training.autoencoder.autoencoder import Autoencoder
from ae_rom_training.autoencoder.baseline_ae import BaselineAE
from ae_rom_training.time_stepper.generic_recurrent import GenericRecurrent


class GenericRecurrentAETS(AEROM):
    def __init__(self, net_idx, input_dict, mllib, network_suffix):

        # get time stepper
        if input_dict["train_ts"]:
            self.time_stepper = GenericRecurrent(net_idx, mllib)
        else:
            self.time_stepper = None

        # get autoencoder
        if input_dict["train_ae"]:
            if input_dict["train_ts"]:
                self.autoencoder = BaselineAE(net_idx, mllib)
            else:
                self.autoencoder = Autoencoder(net_idx, mllib)
        else:
            self.autoencoder = None

        super().__init__(net_idx, input_dict, mllib, network_suffix)

    def build(self):
        """Assemble singular model object for entire network, if possible"""

        if self.training_ae:
            if self.training_ts:
                # will have to run custom training
                return
            else:
                # will run builtin training on autoencoder
                self.model_obj = self.autoencoder.model_obj
        else:
            # will run builtin training on time-stepper
            self.model_obj = self.time_stepper.model_obj

    def save(self, model_dir):
        """Save individual networks and losses in AE ROM"""

        # save models
        if self.training_ae:
            self.autoencoder.save(model_dir, self.network_suffix)
        if self.training_ts:
            self.time_stepper.save(model_dir, self.network_suffix)

        # save reconstruction and prediction losses
        if self.training_ae and self.training_ts:
            loss_train_recon_path = os.path.join(model_dir, "loss_train_recon" + self.network_suffix + ".npy")
            loss_train_step_path = os.path.join(model_dir, "loss_train_step" + self.network_suffix + ".npy")
            loss_val_recon_path = os.path.join(model_dir, "loss_val_recon" + self.network_suffix + ".npy")
            loss_val_step_path = os.path.join(model_dir, "loss_val_step" + self.network_suffix + ".npy")
            np.save(loss_train_recon_path, self.loss_train_recon)
            np.save(loss_train_step_path, self.loss_train_step)
            np.save(loss_val_recon_path, self.loss_val_recon)
            np.save(loss_val_step_path, self.loss_val_step)

        # save builtin training and validation losses
        else:

            loss_train_hist_path = os.path.join(
                model_dir, self.train_prefix + "loss_train_hist" + self.network_suffix + ".npy"
            )
            loss_val_hist_path = os.path.join(
                model_dir, self.train_prefix + "loss_val_hist" + self.network_suffix + ".npy"
            )
            np.save(loss_train_hist_path, self.loss_train_hist)
            np.save(loss_val_hist_path, self.loss_val_hist)

    def train_model_builtin(
        self, data_list_train, data_list_val, optimizer, loss, options, input_dict, params,
    ):
        """Train time stepper with ML library built-in functions.

        Here, data_list_train, data_list_val contain the standardized, encoded latent variables
        """

        # set up time-stepper latent variable data
        if self.training_ts:

            seq_lookback = params["seq_lookback"]
            seq_step = params["seq_step"]
            pred_length = params["pred_length"]

            # if using "random", need to unshuffle, make sequences, then separate sequences out again
            # TODO: this doesn't work with separate validation sets
            # put consecutive data back together again
            data_list_full = []
            for idx, data_train in enumerate(data_list_train):
                data_val = data_list_val[idx]
                num_snaps = data_train.shape[0] + data_val.shape[0]
                split_idxs_train = input_dict["split_idxs_train"][idx]
                split_idxs_val = input_dict["split_idxs_val"][idx]
                data_full = np.zeros((num_snaps,) + data_train.shape[1:], dtype=data_train.dtype)
                data_full[split_idxs_train, ...] = data_train.copy()
                data_full[split_idxs_val, ...] = data_val.copy()
                data_list_full.append(data_full.copy())

            # window data, making inputs and labels
            latent_vars_list_seqs, latent_vars_list_seqs_pred = window(
                data_list_full, seq_lookback, pred_length=pred_length, seq_step=seq_step
            )

            # redistribute training and validation data
            latent_vars_list_train_seqs = []
            latent_vars_list_val_seqs = []
            latent_vars_list_train_seqs_pred = []
            latent_vars_list_val_seqs_pred = []
            for idx, latent_vars_seqs in enumerate(latent_vars_list_seqs):
                latent_vars_seqs_pred = latent_vars_list_seqs_pred[idx]
                # need to exclude 0:seq_lookback, subtract seq_lookback to get new sorting indices
                # NOTE: assume_unique maintains shuffle
                idxs_train = (
                    np.setdiff1d(input_dict["split_idxs_train"][idx], np.arange(0, seq_lookback), assume_unique=True)
                    - seq_lookback
                )
                idxs_val = (
                    np.setdiff1d(input_dict["split_idxs_val"][idx], np.arange(0, seq_lookback), assume_unique=True)
                    - seq_lookback
                )
                latent_vars_list_train_seqs.append(latent_vars_seqs[idxs_train, ...])
                latent_vars_list_train_seqs_pred.append(latent_vars_seqs_pred[idxs_train, ...])
                latent_vars_list_val_seqs.append(latent_vars_seqs[idxs_val, ...])
                latent_vars_list_val_seqs_pred.append(latent_vars_seqs_pred[idxs_val, ...])

            # concatenate and shuffle windowed training data
            data_train_input = np.concatenate(latent_vars_list_train_seqs, axis=0)
            data_val_input = np.concatenate(latent_vars_list_val_seqs, axis=0)
            data_train_output = np.concatenate(latent_vars_list_train_seqs_pred, axis=0)
            data_val_output = np.concatenate(latent_vars_list_val_seqs_pred, axis=0)
            shuffle_idxs = np.random.permutation(np.arange(data_train_input.shape[0]))
            data_train_input = data_train_input[shuffle_idxs, ...]
            data_train_output = data_train_output[shuffle_idxs, ...]

            # squeeze singleton dimensions
            # TODO: might not always be applicable
            data_train_output = np.squeeze(data_train_output, axis=1)
            data_val_output = np.squeeze(data_val_output, axis=1)

        # set up autoencoder data
        else:

            # concatenate and shuffle data sets, since order doesn't matter
            data_train_input = np.concatenate(data_list_train, axis=0)
            data_val_input = np.concatenate(data_list_val, axis=0)
            np.random.shuffle(data_train_input)

            # make data "labels"
            data_train_output = data_train_input
            data_val_output = data_val_input

        # train
        self.loss_train_hist, self.loss_val_hist, loss_train, loss_val = self.mllib.train_model_builtin(
            self.model_obj,
            data_train_input,
            data_train_output,
            data_val_input,
            data_val_output,
            optimizer,
            loss,
            options,
            input_dict,
            params,
        )

        return loss_train, loss_val

    def train_model_custom(
        self, data_list_train, data_list_val, optimizer, loss, options, input_dict, params,
    ):
        """Call custom training loop after organizing data"""

        # get solution time series windows
        seq_length = params["seq_length"]
        seq_step = params["seq_step"]
        seq_lookback = params["seq_lookback"]
        assert seq_lookback < seq_length, (
            "seq_lookback ("
            + str(seq_lookback)
            + ") must not be greater than or equal to seq_length ("
            + str(seq_length)
            + ")"
        )
        data_list_train_seqs = hankelize(data_list_train, seq_length, seq_step=seq_step)
        data_list_val_seqs = hankelize(data_list_val, seq_length, seq_step=seq_step)

        # concatenate; shuffle training sequences
        data_train_seqs = np.concatenate(data_list_train_seqs, axis=0)
        data_val_seqs = np.concatenate(data_list_val_seqs, axis=0)
        shuffle_idxs = np.random.permutation(np.arange(data_train_seqs.shape[0]))
        data_train_seqs = data_train_seqs[shuffle_idxs, ...]

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
            lookback=seq_lookback,
        )

        self.loss_train_recon = loss_addtl_train_list[0]
        self.loss_train_step = loss_addtl_train_list[1]
        self.loss_val_recon = loss_addtl_val_list[0]
        self.loss_val_step = loss_addtl_val_list[1]

        loss_train = loss_train_hist[-1]
        loss_val = loss_val_hist[-1]

        return loss_train, loss_val
