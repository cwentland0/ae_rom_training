
from ae_rom_training.ml_model.ml_model import MLModel

class Encoder(MLModel):

    def __init__(self, input_dict, mllib):

        self.param_prefix = "encoder"
        
        super().__init__(input_dict, mllib)

        

    def preproc_inputs(self, input_dict):
        """"Build assembly instructions"""

        # have to ensure final layer has latent dimension
        encoder_keys = [key for key, value in input_dict.items() if "encoder_" in key]
        self.num_layers = len(input_dict["encoder_layer_types"])

        breakpoint()
        # loop through all inputs

            # if using HyperOpt
                # if input is list
                    # if list of lists
                        # assert that length of list is number of layers
                        # if expression type is list
                            # assert list is number of layers
                        # else
                            # expand expression type
                        # flag this parameter as having layer-wise HyperOpt
                        # loop entries of list
                            # if list
                                # treat sublist as HyperOpt expression definitions
                            # else
                                # treat single values as fixed, do error checking
                        # create list of flags denoting which layers have HyperOpt definitions
                        # add new entries to HyperOpt space with numbered index suffixes and HyperOpt definition
                    # else
                        # if expression type is present
                            # treat input as HyperOpt expression definition
                        # if no expression type is present
                            # treat input as fixed input, assert that length is number of layers

                # else
                    # if expression type is present
                        # throw error, even if "choice" is requested it's pointless and confusing
                    # if no expression type is present
                        # error checking
                        # expand input to number of layers
                
            # else
                # if input is list
                    # assert length if number of layers
                    # if list of lists
                        # throw error, this isn't HyperOpt
                    # else
                        # do error checking on entries
                # else
                    # error checking
                    # expand input to number of layers


        pass