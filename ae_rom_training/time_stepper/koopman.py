import os

import numpy as np

from ae_rom_training.time_stepper.time_stepper import TimeStepper
from ae_rom_training.ml_model.koopman_discrete import KoopmanDiscrete
from ae_rom_training.ml_model.koopman_continuous import KoopmanContinuous


class Koopman(TimeStepper):
    """Model defining a discrete Koopman operator."""

    def __init__(self, input_dict, mllib, continuous=False):

        self.continuous = continuous
        if self.continuous:
            self.stepper = KoopmanContinuous("koopman", input_dict, mllib)
        else:
            self.stepper = KoopmanDiscrete("koopman", input_dict, mllib)
        super().__init__(mllib)

        self.component_networks = [self.stepper]

    # def calc_dmd(self, data, num_modes):
    #     """Compute DMD matrix of dimension num_modes of given data.

    #     Data is assumed to be in NCHW or NHWC, either way the last channels get flattened.
    #     """

    #     num_snaps = data.shape[0]
    #     data_mat = np.reshape(data, (num_snaps, -1), order="C")

    #     data_0 = data_mat[:-1, :].T
    #     data_1 = data_mat[1:, :].T

    #     u_0, s_0, v_0 = np.linalg.svd(data_0, full_matrices=False)
    #     u_0 = u_0[:, :num_modes]
    #     s_0 = s_0[:num_modes]
    #     v_0 = v_0.conj().T[:, :num_modes]

    #     dmd_op = u_0.conj().T @ data_1 @ v_0

    # def init_from_dmd(self, data, num_modes):
    #     pass

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
        koopman_op_input_shape, koopman_op_output_shape = self.mllib.get_layer_io_shape(self.stepper.model_obj, -1)

        # check shapes
        latent_shape = (input_dict["latent_dim"],)
        assert koopman_input_shape == latent_shape, (
            "Koopman input shape does not match latent shape: " + str(koopman_input_shape) + " vs. " + str(latent_shape)
        )

        assert koopman_op_output_shape == latent_shape, (
            "Koopman output shape does not match latent shape: "
            + str(koopman_op_output_shape)
            + " vs. "
            + str(latent_shape)
        )

        # check continuous implementation
        if self.continuous:
            time_input_shape, time_output_shape = self.mllib.get_layer_io_shape(self.stepper.model_obj, 1)
            assert koopman_op_input_shape[0] == latent_shape
            assert koopman_op_input_shape[1] == (1,)
            assert time_input_shape == (1,), "Something went wrong in providing time input to continuous Koopman layer"
            assert time_output_shape == (1,), "Something went wrong in providing time input to continuous Koopman layer"

        print("\nKOOPMAN passed I/O checks!")

    def save(self, model_dir, network_suffix):

        koopman = self.stepper.get_koopman()
        koopman_path = os.path.join(model_dir, "koopman" + network_suffix + ".npy")
        np.save(koopman_path, koopman)
