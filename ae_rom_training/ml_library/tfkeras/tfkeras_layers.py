import tensorflow as tf
from tensorflow.keras.layers import Layer
from ae_rom_training.constants import RANDOM_SEED


class ContinuousKoopman(Layer):
    def __init__(self, units, stable=False, **kwargs):

        super().__init__(**kwargs)

        self.units = units
        self.stable = stable

    def build(self, input_shape):

        assert input_shape.rank == 2, "Input to continuous Koopman is not two-dimensional (batch, units)"
        assert input_shape[-1] == self.units, (
            "Input to continuous Koopman does not have " + str(self.units) + " units, as expected"
        )

        if self.stable:
            self.diag = self.add_weight(
                shape=(self.units,), initializer="glorot_uniform", dtype=self.dtype, trainable=True
            )
            self.off_diags = self.add_weight(
                shape=(self.units - 1,), initializer="glorot_uniform", dtype=self.dtype, trainable=True
            )
            self.set_tridiag()

        else:

            self.K = self.add_weight(shape=(self.units, self.units), initializer="glorot_uniform", trainable=True)

    def set_tridiag(self):
        self.K = (
            tf.linalg.diag(-tf.math.square(self.diag), k=0, num_rows=self.units, num_cols=self.units)
            + tf.linalg.diag(self.off_diags, k=1, num_rows=self.units, num_cols=self.units)
            + tf.linalg.diag(-self.off_diags, k=-1, num_rows=self.units, num_cols=self.units)
        )

    def get_config(self):
        config = super().get_config()
        config.update({"units": self.units})

    def call(self, inputs, Dt):
        """Computes continuous Koopman operation, inputs @ exp(Dt * K)
        
        inputs: tensor of shape (units,) representing initial latent variable value
        Dt: scalar, time step from initial latent variable to desired output latent variable prediction
        """

        if self.stable:
            self.set_tridiag()

        return tf.matmul(inputs, tf.linalg.expm(Dt * self.K))
