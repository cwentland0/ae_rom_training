import os

from ae_rom_training.ml_model.autoencoder.autoencoder import Autoencoder
from ae_rom_training.ml_model.encoder import Encoder
from ae_rom_training.ml_model.decoder import Decoder


class BaselineAE(Autoencoder):
    """Simple autoencoder, only encoder and decoder"""

    def __init__(self, input_dict, mllib, network_suffix):

        self.encoder = Encoder("encoder", mllib)
        self.decoder = Decoder("decoder", mllib)
        self.component_networks = [self.encoder, self.decoder]

        super().__init__(input_dict, mllib, network_suffix)

    def build(self, input_dict, params, data_shape, batch_size=None):

        # assemble encoder
        self.encoder.assemble(input_dict, params, data_shape, batch_size=batch_size)
        self.mllib.display_model_summary(self.encoder.model_obj, displaystr="ENCODER")

        # assemble decoder (mirror, if requested)
        if input_dict["mirrored_decoder"]:
            self.decoder.mirror_encoder(self.encoder, input_dict)
        self.decoder.assemble(input_dict, params, input_dict["latent_dim"])
        self.mllib.display_model_summary(self.decoder.model_obj, displaystr="DECODER")

        # put the two together
        self.model_obj = self.mllib.merge_models([self.encoder.model_obj, self.decoder.model_obj])

    def check_build(self, input_dict, data_shape):
        """Check that autoencoder built ''correctly''

        All this can really do is check that the I/O shapes are as expected.
        """

        # get I/O shapes
        encoder_input_shape, _ = self.mllib.get_layer_io_shape(self.encoder.model_obj, 0)
        _, encoder_output_shape = self.mllib.get_layer_io_shape(self.encoder.model_obj, -1)
        decoder_input_shape, _ = self.mllib.get_layer_io_shape(self.decoder.model_obj, 0)
        _, decoder_output_shape = self.mllib.get_layer_io_shape(self.decoder.model_obj, -1)

        # check shapes
        latent_shape = (input_dict["latent_dim"],)
        assert encoder_input_shape == data_shape, (
            "Encoder input shape does not match data shape: " + str(encoder_input_shape) + " vs. " + str(data_shape)
        )
        assert encoder_output_shape == latent_shape, (
            "Encoder output shape does not match latent shape: "
            + str(encoder_output_shape)
            + " vs. "
            + str(latent_shape)
        )
        assert decoder_input_shape == latent_shape, (
            "Decoder input shape does not match latent shape: " + str(decoder_input_shape) + " vs. " + str(latent_shape)
        )
        assert decoder_output_shape == data_shape, (
            "Decoder output shape does not match data shape: " + str(decoder_output_shape) + " vs. " + str(data_shape)
        )

    def train(self, input_dict, params, data_train, data_val):

        # get training parameters
        loss = self.mllib.get_loss_function(input_dict["loss_func"], params)
        optimizer = self.mllib.get_optimizer(input_dict["optimizer"], params)
        options = self.mllib.get_options(input_dict, params)

        # compile and train
        self.mllib.compile_model(self.model_obj, optimizer, loss)
        self.mllib.train_model(self.model_obj, params, data_train, data_train, data_val, data_val, options)

        # report training and validation loss
        loss_train = self.mllib.calc_loss(self.model_obj, data_train, data_train)
        loss_val = self.mllib.calc_loss(self.model_obj, data_val, data_val)

        # pull out encoder and decoder models
        # TODO: this layer indexing may not be valid for PyTorch
        self.encoder.model_obj = self.mllib.get_layer(self.model_obj, -2)
        self.decoder.model_obj = self.mllib.get_layer(self.model_obj, -1)

        return loss_train, loss_val

    def save(self, model_dir):

        encoder_path = os.path.join(model_dir, "encoder" + self.network_suffix)
        self.mllib.save_model(self.encoder.model_obj, encoder_path)

        decoder_path = os.path.join(model_dir, "decoder" + self.network_suffix)
        self.mllib.save_model(self.decoder.model_obj, decoder_path)
