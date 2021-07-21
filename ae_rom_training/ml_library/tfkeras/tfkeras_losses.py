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


class AETSCombinedLoss(LossFunctionWrapper):
    """Wrapper class for computing the combined autoencoder/time stepper loss"""

    def __init__(
        self,
        reduction=losses_utils.ReductionV2.AUTO,
        name="ae_ts_combined_loss",
        normalize=False,
        eps=1e-12,
        alpha=1.0,
        beta=1.0,
    ):

        super().__init__(
            ae_ts_combined_error, reduction=reduction, name=name, normalize=normalize, eps=eps, alpha=alpha, beta=beta
        )


def ae_ts_combined_error(
    y_low_true, y_low_pred, y_full_true, y_full_pred, normalize=False, eps=1e-12, alpha=1.0, beta=1.0,
):
    """Simple loss for prediction and reconstruction error of combined AE/TS.
    
    y_low_true is the "true" low-dimensional state encoded from the full-dimensional state
    y_low_pred is the predicted low-dimensional state advanced via the time-stepper

    y_full_true is the true full-dimensional state
    y_full_pred is the predicted full-dimensional state via decoding y_low_pred

    alpha scales the reconstruction loss, beta scales the time-step loss

    All input tensors are of shape [num_snaps, num_dof]
    """

    recon_loss_vec = tf.norm(tf.math.subtract(y_full_true, y_full_pred), ord=2, axis=1)
    step_loss_vec = tf.norm(tf.math.subtract(y_low_true, y_low_pred), ord=2, axis=1)

    if normalize:
        recon_loss_norm_vec = tf.add(tf.norm(y_full_true, ord=2, axis=1), eps)
        step_loss_norm_vec = tf.add(tf.norm(y_low_true, ord=2, axis=1), eps)

        recon_loss_vec = tf.math.divide(recon_loss_vec, recon_loss_norm_vec)
        step_loss_vec = tf.math.divide(step_loss_vec, step_loss_norm_vec)

    recon_loss = tf.math.reduce_mean(recon_loss_vec)
    step_loss = tf.math.reduce_mean(step_loss_vec)

    return alpha * recon_loss + beta * step_loss


# def ae_ts_combined_norm():
# """Normalized version of combined autoencoder/time stepper loss"""
