import tensorflow as tf
from tensorflow.python.keras.losses import MeanSquaredError, LossFunctionWrapper
import tensorflow.python.keras.utils.losses_utils as losses_utils


def pure_l2(y_true, y_pred):
    """Strict L2 error"""

    return tf.math.reduce_sum(tf.math.square(y_true - y_pred))


def pure_mse(y_true, y_pred):
    """Strict mean-squared error, not including regularization contribution"""

    mse = MeanSquaredError()
    return mse(y_true, y_pred)

@tf.function
def ae_ts_combined_error(data_seq, data_seq_ignore, ae_rom):
    """Simple loss for prediction and reconstruction error of combined AE/TS."""

    # TODO: ignore the second input, it's not relevant

    loss_recon = 0.0
    loss_step = 0.0

    # initial encoding
    latent_vars_pred_init = ae_rom.autoencoder.encoder.model_obj(data_seq[:, 0, ...])
    sol_pred_init = ae_rom.autoencoder.decoder.model_obj(latent_vars_pred_init)

    sol_seq = tf.cast(data_seq, sol_pred_init.dtype)
    seq_length = sol_seq.shape[1]

    # initial reconstruction loss (no advancing with time stepper)
    loss_recon += tf.math.reduce_mean(tf.math.squared_difference(sol_pred_init, sol_seq[:, 0, ...]))

    latent_vars_pred = tf.identity(latent_vars_pred_init)

    for seq in range(1, seq_length - 1):

        # encode solution
        latent_vars_encode = ae_rom.autoencoder.encoder.model_obj(sol_seq[:, seq, ...])
        
        # make time stepper prediction
        latent_vars_pred = ae_rom.time_stepper.stepper.model_obj(latent_vars_pred)

        # decode predicted solution
        sol_pred = ae_rom.autoencoder.decoder.model_obj(latent_vars_pred)

        loss_recon += tf.math.reduce_mean(tf.math.squared_difference(sol_pred, sol_seq[:, seq + 1, ...]))
        loss_step += tf.math.reduce_mean(tf.math.squared_difference(latent_vars_pred, latent_vars_encode))

    return [loss_recon + loss_step, loss_recon, loss_step]

