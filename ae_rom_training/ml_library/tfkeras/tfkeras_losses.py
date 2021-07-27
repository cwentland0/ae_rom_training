import tensorflow as tf
from tensorflow.python.keras.losses import MeanSquaredError, LossFunctionWrapper
import tensorflow.python.keras.utils.losses_utils as losses_utils

import matplotlib.pyplot as plt


def pure_l2(y_true, y_pred):
    """Strict L2 error"""

    return tf.math.reduce_sum(tf.math.square(y_true - y_pred))


def pure_mse(y_true, y_pred):
    """Strict mean-squared error, not including regularization contribution"""

    mse = MeanSquaredError()
    return mse(y_true, y_pred)


@tf.function
def ae_ts_combined_error(data_seq, data_seq_ignore, ae_rom, lookback=1, continuous=False, step_sizes=None):
    """Simple loss for prediction and reconstruction error of combined AE/TS."""

    seq_length = data_seq.shape[1]
    if continuous:
        assert step_sizes is not None, "If making continuous predictions, must provide step_sizes"
        assert step_sizes.shape[0] == (seq_length - lookback + 1), (
            "step_sizes length mismatch: is "
            + str(step_sizes.shape[0])
            + ", should be "
            + str(seq_length - lookback + 1)
        )

    # TODO: ignore data_seq_ignore, it's not relevant

    loss_recon = 0.0
    loss_step = 0.0

    # lookback window encoding
    for seq in range(0, lookback):
        latent_vars_encode = ae_rom.autoencoder.encoder.model_obj(data_seq[:, seq, ...])
        sol_decode = ae_rom.autoencoder.decoder.model_obj(latent_vars_encode)

        # initial reconstruction loss (no advancing with time stepper)
        if seq == 0:
            sol_seq = tf.cast(data_seq, sol_decode.dtype)
        loss_recon += tf.math.reduce_mean(tf.math.squared_difference(sol_decode, sol_seq[:, seq, ...]))

        if lookback > 1:
            latent_vars_encode = tf.expand_dims(latent_vars_encode, axis=1)

        if seq == 0:
            latent_vars_lookback = tf.identity(latent_vars_encode)
        else:
            latent_vars_lookback = tf.concat([latent_vars_lookback, latent_vars_encode], axis=1)

    for seq in range(lookback, seq_length):

        # encode solution
        latent_vars_encode = ae_rom.autoencoder.encoder.model_obj(sol_seq[:, seq, ...])

        # make time stepper prediction
        latent_vars_pred = ae_rom.time_stepper.stepper.model_obj(latent_vars_lookback)

        # decode predicted solution
        sol_pred = ae_rom.autoencoder.decoder.model_obj(latent_vars_pred)

        loss_recon += tf.math.reduce_mean(tf.math.squared_difference(sol_pred, sol_seq[:, seq, ...]))
        loss_step += tf.math.reduce_mean(tf.math.squared_difference(latent_vars_pred, latent_vars_encode))

        # update lookback window
        if lookback == 1:
            latent_vars_lookback = latent_vars_pred
        else:
            latent_vars_lookback = tf.concat(
                [latent_vars_lookback[:, 1:, ...], tf.expand_dims(latent_vars_pred, axis=1)]
            )

    loss_recon /= seq_length
    loss_step /= seq_length - lookback + 1

    return [loss_recon + loss_step, loss_recon, loss_step]
