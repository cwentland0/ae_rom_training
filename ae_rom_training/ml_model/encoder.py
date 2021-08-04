from ae_rom_training.ml_model.ml_model import MLModel


class Encoder(MLModel):
    def __init__(self, net_idx, param_prefix, mllib):

        super().__init__(net_idx, param_prefix, mllib)
