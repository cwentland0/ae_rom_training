from ae_rom_training.ae_rom.ae_rom import AEROM
from ae_rom_training.autoencoder.baseline_ae import BaselineAE


class BaselineAEROM(AEROM):
    """Simple AE ROM with only autoencoder (no time-stepper)"""

    def __init__(self, input_dict, mllib, network_suffix):

        self.autoencoder = BaselineAE(mllib)
        self.time_stepper = None

        super().__init__(input_dict, mllib, network_suffix)
