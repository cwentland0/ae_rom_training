from ae_rom_training.ml_model.ml_model import MLModel

class Decoder(MLModel):

    def __init__(self, input_dict, mllib):

        self.param_prefix = "decoder"
        super().__init__(input_dict, mllib)

        

    def preproc_inputs():
        """"Build assembly instructions"""
        pass

    # def preproc_layer_input_mirror():
    #     """Build assembly instructions by reversing encoder assembly instructions"""