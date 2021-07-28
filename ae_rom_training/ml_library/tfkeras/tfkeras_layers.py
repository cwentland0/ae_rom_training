import tensorflow as tf
from tensorflow.keras.layers import Layer
from ae_rom_training.constants import RANDOM_SEED


class ContinuousKoopman(Layer):
    def __init__(self, output_size, stable=False, kernel_initializer="glorot_uniform", **kwargs):

        super().__init__(**kwargs)

        self.output_size = output_size
        self.stable = stable
        self.kernel_initializer = kernel_initializer

    def build(self, input_shape):

        assert (input_shape[0].rank == 2) and (input_shape[1].rank == 2), (
            "Input to continuous Koopman is not two-dimensional (batch, output_size)"
        )
        assert input_shape[0][-1] == self.output_size, (
            "Input to continuous Koopman does not have " + str(self.output_size) + " units, as expected"
        )
        assert input_shape[1][-1] == 1, (
            "Time input to continuous Koopman does not have 1 unit, as expected"
        )

        if self.stable:
            self.diag = self.add_weight(
                shape=(self.output_size,), initializer=self.kernel_initializer, dtype=self.dtype, trainable=True
            )
            self.off_diags = self.add_weight(
                shape=(self.output_size - 1,), initializer=self.kernel_initializer, dtype=self.dtype, trainable=True
            )
            self.set_tridiag()

        else:

            self.K = self.add_weight(shape=(self.output_size, self.output_size), initializer=self.kernel_initializer, trainable=True)

    def set_tridiag(self):
        self.K = (
            tf.linalg.diag(-tf.math.square(self.diag), k=0, num_rows=self.output_size, num_cols=self.output_size)
            + tf.linalg.diag(self.off_diags, k=1, num_rows=self.output_size, num_cols=self.output_size)
            + tf.linalg.diag(-self.off_diags, k=-1, num_rows=self.output_size, num_cols=self.output_size)
        )

    def get_koopman_numpy(self):

        if self.stable:
            self.set_tridiag()

        return self.K.numpy()

    def get_config(self):
        config = super().get_config()
        config.update({"output_size": self.output_size, "stable": self.stable, "kernel_initializer": self.kernel_initializer})

    def call(self, inputs):
        """Computes continuous Koopman operation, inputs @ exp(Dt * K)
        
        inputs: tensor of shape (output_size,) representing initial latent variable value
        Dt: scalar, time step of desired output latent variable prediction
        """

        if self.stable:
            self.set_tridiag()

        # compute continuous Koopman operator 
        exp = tf.linalg.expm(tf.expand_dims(inputs[1], axis=-1) * tf.expand_dims(self.K, axis=0))

        # compute prediction
        pred = tf.matmul(tf.expand_dims(inputs[0], axis=1), exp)

        return tf.squeeze(pred, axis=1)
