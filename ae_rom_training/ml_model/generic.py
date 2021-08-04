from ae_rom_training.ml_model.ml_model import MLModel


class Generic(MLModel):
    """Generic model that can be built from sequential stock layers"""

    def __init__(self, net_idx, param_prefix, mllib):
        super().__init__(net_idx, param_prefix, mllib)
