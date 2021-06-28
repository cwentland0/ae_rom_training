
from ae_rom_training.ml_model.ml_model import MLModel

class Encoder(MLModel):

    def __init__(self, input_dict, param_space, mllib):

        self.param_prefix = "encoder"
        
        super().__init__(input_dict, param_space, mllib)
        