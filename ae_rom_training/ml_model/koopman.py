from ae_rom_training.ml_model.ml_model import MLModel

# TODO:
# 1) Initialize Koopman with DMD Koopman
# 2) 

class Koopman(MLModel):
    """Model defining a linear Koopman operator.
    
    May be a linear or discrete Koopman.
    """

    def __init__(self, param_prefix, input_dict, param_space, mllib):

        super().__init__(param_prefix, input_dict, param_space, mllib)


    def calc_discrete_koopman(self):
        pass

    def calc_continuous_koopman(self):
        pass