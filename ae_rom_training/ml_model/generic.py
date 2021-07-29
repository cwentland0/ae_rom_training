from ae_rom_training.ml_model.ml_model import MLModel


class Generic(MLModel):
    """Generic model that can be built from sequential stock layers"""

    def __init__(self, param_prefix, mllib):
        super().__init__(param_prefix, mllib)
