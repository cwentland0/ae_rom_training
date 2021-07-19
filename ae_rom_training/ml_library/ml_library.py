from ae_rom_training.preproc_utils import catch_input


class MLLibrary:
    """Base class for machine learning library-specific functionality"""

    def __init__(self, input_dict):

        run_gpu = catch_input(input_dict, "run_gpu", False)
        self.init_gpu(run_gpu)

    def train_model():
        pass

    def save_model():
        pass
