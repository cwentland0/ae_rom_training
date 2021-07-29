import os

from ae_rom_training.time_stepper.time_stepper import TimeStepper
from ae_rom_training.ml_model.generic import Generic


class GenericRecurrent(TimeStepper):
    """Recurrent time stepper that can be built from sequential stock layers and trained on windowed data.

    Recurrent layers include GRU, LSTM, TCN, etc.
    E.g. LSTM can just be built from stacked LSTM layers.
    All layers must be defined by user.
    """

    def __init__(self, mllib):

        # TODO: make assertions about return_sequence for stacked recurrent layers

        self.stepper = Generic("stepper", mllib)

        super().__init__(mllib)

        self.component_networks = [self.stepper]

    def build(self, input_dict, params, batch_size=None):

        # assemble generic sequential model
        self.stepper.assemble(
            input_dict, params, (params["seq_lookback"], input_dict["latent_dim"],), batch_size=batch_size
        )
        self.mllib.display_model_summary(self.stepper.model_obj, displaystr="STEPPER")
        self.model_obj = self.stepper.model_obj

    def check_build(self, input_dict, params):
        """Check that stepper built correctly.

        Assumes that input has shape (seq_lookback, latent_dims,), and output has shape (latent_dims,).
        """

        # get I/O shapes
        input_shape, _ = self.mllib.get_layer_io_shape(self.model_obj, 0)
        _, output_shape = self.mllib.get_layer_io_shape(self.model_obj, -1)

        # check shapes
        exp_input_shape = (
            params["seq_lookback"],
            input_dict["latent_dim"],
        )
        exp_output_shape = (input_dict["latent_dim"],)
        assert input_shape == exp_input_shape, (
            "Stepper input shape does not match latent shape: " + str(input_shape) + " vs. " + str(exp_input_shape)
        )
        assert output_shape == exp_output_shape, (
            "Stepper output shape does not match latent shape: " + str(output_shape) + " vs. " + str(exp_output_shape)
        )

        print("\nSTEPPER passed I/O checks!")

    def save(self, model_dir, network_suffix):

        stepper_path = os.path.join(model_dir, "stepper" + network_suffix)
        self.mllib.save_model(self.model_obj, stepper_path)
