from ae_rom_training.ml_model.ml_model import MLModel


class KoopmanContinuous(MLModel):
    """Continuous Koopman operator model.
    
    Computes advance from initial state via x(t_0) exp(t_{n+1} * K). 
    """

    def __init__(self, param_prefix, input_dict, mllib):

        input_dict[param_prefix + "_layer_type"] = ["koopman_continuous"]
        input_dict[param_prefix + "_output_size"] = [input_dict["latent_dim"]]
        input_dict[param_prefix + "_layer_input_idx"] = [-1]
        input_dict[param_prefix + "_use_bias"] = False
        input_dict[param_prefix + "_activation"] = "linear"
        input_dict[param_prefix + "_kern_reg"] = None
        input_dict[param_prefix + "_kern_reg_val"] = 0.0
        input_dict[param_prefix + "_act_reg"] = None
        input_dict[param_prefix + "_act_reg_val"] = 0.0
        input_dict[param_prefix + "_bias_reg"] = None
        input_dict[param_prefix + "_bias_reg_val"] = 0.0
        input_dict[param_prefix + "_bias_init"] = None

        # Check if initializing by DMD. If so, just initialize with Glorot for now, handle later
        self.init_dmd = False
        if param_prefix + "_kern_init" in input_dict:
            kern_init_input = input_dict[param_prefix + "_kern_init"]
            if isinstance(kern_init_input, list):
                if kern_init_input[0] == "dmd":
                    self.init_dmd = True
            elif isinstance(kern_init_input, str):
                if kern_init_input == "dmd":
                    self.init_dmd = True
        if self.init_dmd:
            input_dict[param_prefix + "_kern_init"] = ["glorot_uniform"]

        super().__init__(param_prefix, mllib)

    def get_koopman(self):
        """Retrieve linear operator from model object."""

        return self.mllib.get_koopman(self.model_obj)
