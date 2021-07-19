from ae_rom_training.autoencoder.autoencoder import Autoencoder


class BaselineAE(Autoencoder):
    """Simple autoencoder, only encoder and decoder"""

    def __init__(self, mllib):

        super().__init__(mllib)

        # no extra networks are added

    def build(self, input_dict, params, data_shape, batch_size=None):

        # build encoder and decoder
        super().build(input_dict, params, data_shape, batch_size=batch_size)

        # put the two together
        self.model_obj = self.mllib.merge_models([self.encoder.model_obj, self.decoder.model_obj])

    def check_build(self, input_dict, data_shape):
        """Check that autoencoder built ''correctly''"""

        super().check_build(input_dict, data_shape)

        # nothing else to check besides encoder and decoder

    def train(self, input_dict, params, data_train, data_val):
        """Trains the autoencoder alone."""

        loss_train, loss_val = super().train(input_dict, params, data_train, data_val)

        # pull out encoder and decoder models
        # TODO: this layer indexing may not be valid for PyTorch
        self.encoder.model_obj = self.mllib.get_layer(self.model_obj, -2)
        self.decoder.model_obj = self.mllib.get_layer(self.model_obj, -1)

        return loss_train, loss_val

    def save(self, model_dir):
        """Save autoencoder component networks."""

        super().save(model_dir)

        # nothing else to save besides encoder and decoder
