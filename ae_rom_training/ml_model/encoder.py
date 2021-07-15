
from ae_rom_training.ml_model.ml_model import MLModel

class Encoder(MLModel):

    def __init__(self, param_prefix, input_dict, param_space, mllib):
        
        super().__init__(param_prefix, input_dict, param_space, mllib)
        