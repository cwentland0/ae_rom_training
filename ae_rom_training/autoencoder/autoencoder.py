import os

from ae_rom_training.ml_model.encoder import Encoder
from ae_rom_training.ml_model.decoder import Decoder


class Autoencoder:
    """Base class for autoencoders.
    
    All autoencoders have an outer "encoder" and a "decoder".
    Child classes implement additional component networks and features.
    """

    def __init__(self, mllib):

        self.mllib = mllib
        self.encoder = Encoder("encoder", mllib)
        self.decoder = Decoder("decoder", mllib)
        self.component_networks = [self.encoder, self.decoder]

    def build(self, input_dict, params, data_shape, batch_size=None):
        """Builds required outer encoder and decoder.
        
        Child class implementations should build any additional component networks.
        """

        # assemble encoder
        self.encoder.assemble(input_dict, params, data_shape, batch_size=batch_size)
        self.mllib.display_model_summary(self.encoder.model_obj, displaystr="ENCODER")

        # assemble decoder (mirror, if requested)
        if input_dict["mirrored_decoder"]:
            self.decoder.mirror_encoder(self.encoder, input_dict)
        self.decoder.assemble(input_dict, params, input_dict["latent_dim"])
        self.mllib.display_model_summary(self.decoder.model_obj, displaystr="DECODER")

    def check_build(self, input_dict, data_shape):
        """Check that outer encoder and decoder built ''correctly''
        
        Child class implementations should check any additional component networks.
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
        """Trains the autoencoder alone.
        
        Child class implementations should return losses and pull out trained component networks.
        """

        # get training parameters
        loss = self.mllib.get_loss_function(input_dict["ae_loss_func"])
        optimizer = self.mllib.get_optimizer(input_dict["ae_optimizer"], learn_rate=params["ae_learn_rate"])

        # TODO: this is kind of jank
        if "ae_early_stopping" in input_dict:
            early_stopping = input_dict["ae_early_stopping"]
            if input_dict["ae_early_stopping"]:
                es_patience = input_dict["ae_es_patience"]
            else:
                es_patience = None
        else:
            early_stopping = False
            es_patience = None
        options = self.mllib.get_options(early_stopping, es_patience)

        # compile and train
        self.mllib.compile_model(self.model_obj, optimizer, loss)
        self.mllib.train_model(
            self.model_obj,
            params["ae_batch_size"],
            params["ae_max_epochs"],
            data_train,
            data_train,
            data_val,
            data_val,
            options,
        )

        # report training and validation loss
        loss_train = self.mllib.calc_loss(self.model_obj, data_train, data_train)
        loss_val = self.mllib.calc_loss(self.model_obj, data_val, data_val)

        return loss_train, loss_val

    def save(self, model_dir):
        """Save encoder and decoder models.
        
        Child class implementations should save any additional component network models.
        """

        encoder_path = os.path.join(model_dir, "encoder" + self.network_suffix)
        self.mllib.save_model(self.encoder.model_obj, encoder_path)

        decoder_path = os.path.join(model_dir, "decoder" + self.network_suffix)
        self.mllib.save_model(self.decoder.model_obj, decoder_path)
