import numpy as np

from ae_rom_training.time_stepper.time_stepper import TimeStepper
from ae_rom_training.ml_model.koopman import Koopman

class KoopmanDiscrete(TimeStepper):
    """Model defining a discrete Koopman operator."""

    def __init__(self, input_dict, mllib):

        self.stepper = Koopman("koopman", input_dict, mllib)
        super().__init__(mllib)

        self.component_networks = [self.stepper]

    def calc_dmd(self, data, num_modes):
        """Compute DMD matrix of dimension num_modes of given data.
        
        Data is assumed to be in NCHW or NHWC, either way the last channels get flattened.
        """

        num_snaps = data.shape[0]
        data_mat = np.reshape(data, (num_snaps, -1), order="C")

        data_0 = data_mat[:-1, :].T
        data_1 = data_mat[1:, :].T

        u_0, s_0, v_0 = np.linalg.svd(data_0, full_matrices=False)
        u_0 = u_0[:, :num_modes]
        s_0 = s_0[:num_modes]
        v_0 = v_0.conj().T[:, :num_modes]

        dmd_op = u_0.conj().T @ data_1 @ v_0
        breakpoint()

    def init_from_dmd(self, data, num_modes):
        pass

    def extract_operator(self):
        """Extract linear operator from network.""" 

        pass

    def step(self):
        """Steps system forward"""

        pass

    def build(self, input_dict, params, batch_size=None):

        # assemble Koopman layer
        # TODO: batch_size may be inappropriate here?
        self.stepper.assemble(input_dict, params, (input_dict["latent_dim"],), batch_size=batch_size)
        self.mllib.display_model_summary(self.stepper.model_obj, displaystr="KOOPMAN")

        self.model_obj = self.stepper.model_obj
        

    def check_build(self, input_dict):
        """Check that Koopman built ''correctly''
        
        All this can really do is check that the I/O shapes are as expected.
        """

        # get I/O shapes
        koopman_input_shape, _ = self.mllib.get_layer_io_shape(self.stepper.model_obj, 0)
        _, koopman_output_shape = self.mllib.get_layer_io_shape(self.stepper.model_obj, -1)

        # check shapes
        latent_shape = (input_dict["latent_dim"],)
        assert koopman_input_shape == latent_shape, (
            "Koopman input shape does not match latent shape: " + str(koopman_input_shape) + " vs. " + str(latent_shape)
        )
        assert koopman_output_shape == latent_shape, (
            "Koopman output shape does not match latent shape: "
            + str(koopman_output_shape)
            + " vs. "
            + str(latent_shape)
        )
        print("KOOPMAN passed I/O checks!")

    def train(self, input_dict, params, data_train, data_val):

        # custom training loop, don't use fit
        # for epoch in num_epochs:
        #   for batch_iter in batch_size:
        #       compute autoencoding, saving the latent variables at each step
        #       compute Koopman predictions
        #       store delta
        #   compute loss over batch
        #   backwards propagation on Koopman and autoencoder

        # get training parameters
        # loss = self.mllib.get_loss_function(input_dict["loss_func"], params)
        # optimizer = self.mllib.get_optimizer(input_dict["optimizer"], params)
        # options = self.mllib.get_options(input_dict, params)

        # # compile and train
        # self.mllib.compile_model(self.model_obj, optimizer, loss)
        # self.mllib.train_model(self.model_obj, params, data_train, data_train, data_val, data_val, options)

        # # report training and validation loss
        # loss_train = self.mllib.calc_loss(self.model_obj, data_train, data_train)
        # loss_val = self.mllib.calc_loss(self.model_obj, data_val, data_val)

        # # pull out encoder and decoder models
        # # TODO: this layer indexing may not be valid for PyTorch
        # self.encoder.model_obj = self.mllib.get_layer(self.model_obj, -2)
        # self.decoder.model_obj = self.mllib.get_layer(self.model_obj, -1)

        return loss_train, loss_val

    def calc_loss(self, koopman, true_snaps, true_latent, pred_snaps, pred_latent, delta_vec):

        # normalization constants

        # TODO: this should be a Tensorflow function, just writing things out
        ae_loss = (1.0) / (1.0 + self.beta) * ()

        return loss

    def save(self, model_dir):

        encoder_path = os.path.join(model_dir, "encoder" + self.network_suffix)
        self.mllib.save_model(self.encoder.model_obj, encoder_path)

        decoder_path = os.path.join(model_dir, "decoder" + self.network_suffix)
        self.mllib.save_model(self.decoder.model_obj, decoder_path)