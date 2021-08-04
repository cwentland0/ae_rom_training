from ae_rom_training.autoencoder.autoencoder import Autoencoder


class BaselineAE(Autoencoder):
    """Standalone autoencoder, only encoder and decoder.

    The main distinction with parent Autoencoder is the merging of the encoder and decoder for single-model training.
    """

    def __init__(self, net_idx, mllib):

        super().__init__(net_idx, mllib)

        # no extra networks are added to component_networks

    def build(self, input_dict, params, data_shape, ae, ts, network_suffix, batch_size=None):

        # build encoder and decoder
        super().build(input_dict, params, data_shape, ae, ts, network_suffix, batch_size=batch_size)

        # put the two together if not training a time-stepper
        if not ts:
            self.model_obj = self.mllib.merge_models([self.encoder.model_obj, self.decoder.model_obj])

    def check_build(self, input_dict, params, data_shape):
        """Check that autoencoder built ''correctly''"""

        super().check_build(input_dict, params, data_shape)

        # nothing else to check besides encoder and decoder

    def train(self, input_dict, params, data_train, data_val):
        """Trains the autoencoder alone."""

        loss_train, loss_val = super().train(input_dict, params, data_train, data_val)

        # pull out encoder and decoder models
        # TODO: this layer indexing may not be valid for PyTorch
        self.encoder.model_obj = self.mllib.get_layer(self.model_obj, -2)
        self.decoder.model_obj = self.mllib.get_layer(self.model_obj, -1)

        return loss_train, loss_val

    def save(self, model_dir, network_suffix):
        """Save autoencoder component networks."""

        super().save(model_dir, network_suffix)

        # nothing else to save besides encoder and decoder
