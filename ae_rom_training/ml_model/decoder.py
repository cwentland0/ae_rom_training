from ae_rom_training.ml_model.ml_model import MLModel

class Decoder(MLModel):

    def __init__(self, input_dict, param_space, mllib):

        self.param_prefix = "decoder"
        super().__init__(input_dict, param_space, mllib)

        

    def mirror_encoder():
        """Build parameter space by mirroring encoder."""
        pass