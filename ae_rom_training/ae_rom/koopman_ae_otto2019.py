from ae_rom_training.ae_rom.ae_rom import AEROM
from ae_rom_training.autoencoder.baseline_ae import BaselineAE
from ae_rom_training.time_stepper.koopman_discrete import KoopmanDiscrete

class KoopmanAEOtto2019(AEROM):
    """Autoencoder which learns discrete Koopman, via Otto and Rowley (2019)"""

    def __init__(self, input_dict, mllib, network_suffix):

        self.autoencoder = BaselineAE(mllib)
        self.time_stepper = KoopmanDiscrete(input_dict, mllib)

        super().__init__(input_dict, mllib, network_suffix)
