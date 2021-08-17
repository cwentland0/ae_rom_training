from time import time

import tensorflow as tf
from tensorflow.python.keras.losses import MeanSquaredError


def pure_l2(y_true, y_pred):
    """Strict L2 error"""

    return tf.math.reduce_sum(tf.math.square(y_true - y_pred))


def pure_mse(y_true, y_pred):
    """Strict mean-squared error, not including regularization contribution"""

    mse = MeanSquaredError()
    return mse(y_true, y_pred)


@tf.function
def ae_ts_combined_error(data_seq, data_seq_ignore, ae_rom, lookback=1, continuous=False, time_values=None, **kwargs):
    """Simple loss for prediction and reconstruction error of combined AE/TS."""

    seq_length = data_seq.shape[1]
    data_feat_shape = tuple(data_seq.shape[2:])
    if continuous:
        assert lookback == 1, "lookback must be equal to 1 for continuous prediction"
        assert time_values is not None, "If making continuous predictions, must provide time_values"
        assert time_values.shape[1] == seq_length, (
            "step_sizes length mismatch: is " + str(time_values.shape[0]) + ", should be " + str(seq_length)
        )
        time_values_tensor = tf.convert_to_tensor(time_values)

    # TODO: ignore data_seq_ignore, it's not relevant

    loss_recon = 0.0
    loss_step = 0.0

    # encode and decode everything in a big batch
    data_seq_flat = tf.reshape(data_seq, (-1,) + data_feat_shape)
    latent_vars_encode_flat = ae_rom.autoencoder.encoder.model_obj(data_seq_flat)
    latent_vars_feat_shape = tuple(latent_vars_encode_flat.shape[1:])
    latent_vars_encode = tf.reshape(latent_vars_encode_flat, (-1, seq_length,) + latent_vars_feat_shape)
    latent_vars_lookback = latent_vars_encode[:, :lookback, ...]
    latent_vars_lookback_flat = tf.reshape(latent_vars_lookback, (-1,) + latent_vars_feat_shape)
    sol_decode_flat = ae_rom.autoencoder.decoder.model_obj(latent_vars_lookback_flat)
    sol_decode = tf.reshape(sol_decode_flat, (-1, lookback,) + data_feat_shape)
    sol_seq = tf.cast(data_seq, sol_decode.dtype)
    # mean_axes = [tf.TensorShape([0]) + tf.range(2, 2 + len(data_feat_shape))]
    mean_axes = [0] + [i for i in range(2, 2 + len(data_feat_shape))]
    loss_recon = tf.math.reduce_sum(
        tf.math.reduce_mean(tf.math.squared_difference(sol_decode, sol_seq[:, :lookback, ...]), axis=mean_axes)
    )

    # prediction
    for seq in range(lookback, seq_length):

        # make time stepper prediction
        if continuous:
            latent_vars_pred = ae_rom.time_stepper.stepper.model_obj(
                [latent_vars_lookback, time_values_tensor[:, seq, ...]]
            )
        else:
            latent_vars_pred = ae_rom.time_stepper.stepper.model_obj(latent_vars_lookback)

        # decode predicted solution
        sol_pred = ae_rom.autoencoder.decoder.model_obj(latent_vars_pred)

        loss_recon += tf.math.reduce_mean(tf.math.squared_difference(sol_pred, sol_seq[:, seq, ...]))
        loss_step += tf.math.reduce_mean(tf.math.squared_difference(latent_vars_pred, latent_vars_encode[:, seq, ...]))

        # update lookback window
        # don't update for continuous, as latent_vars_loopback should stay equal to the initial sequence state
        if not continuous:
            if lookback == 1:
                latent_vars_lookback = latent_vars_pred
            else:
                if seq < (seq_length - 1):
                    latent_vars_lookback = tf.concat(
                        [latent_vars_lookback[:, 1:, ...], tf.expand_dims(latent_vars_pred, axis=1)], axis=1,
                    )

    loss_recon /= seq_length
    loss_step /= seq_length - lookback

    return [loss_recon + loss_step, loss_recon, loss_step]
