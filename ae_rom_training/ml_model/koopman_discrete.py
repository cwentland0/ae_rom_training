from ae_rom_training.ml_model.ml_model import MLModel


class KoopmanDiscrete(MLModel):
    """Linear Koopman operator.
    
    Extremely simple, just a single linear dense layer with zero bias, no regularization.
    """

    def __init__(self, param_prefix, input_dict, mllib):

        input_dict[param_prefix + "_layer_type"] = ["dense"]
        input_dict[param_prefix + "_layer_input_idx"] = [-1]
        input_dict[param_prefix + "_use_bias"] = False
        input_dict[param_prefix + "_activation"] = "linear"
        input_dict[param_prefix + "_kern_reg"] = None
        input_dict[param_prefix + "_kern_reg_val"] = 0.0
        input_dict[param_prefix + "_act_reg"] = None
        input_dict[param_prefix + "_act_reg_val"] = 0.0
        input_dict[param_prefix + "_bias_reg"] = None
        input_dict[param_prefix + "_bias_reg_val"] = 0.0
        input_dict[param_prefix + "_kern_init"] = None
        input_dict[param_prefix + "_bias_init"] = None
        input_dict[param_prefix + "_output_size"] = input_dict["latent_dim"]

        super().__init__(param_prefix, mllib)

    def get_koopman(self):
        """Retrieve linear operator from model object.
        
        Transpose is returned because Dense layer calculation is computed as
        output = input @ kernel + bias, while we (for aesthetic purposes only)
        want a Koopman operator of the form x^(n+1) = K @ x^(n).
        """

        return self.mllib.get_layer_weights(self.model_obj, 1, weights=True).T
