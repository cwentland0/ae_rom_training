from time import sleep

from ae_rom_training.ml_model.ml_model import MLModel

class Decoder(MLModel):

    def __init__(self, input_dict, param_space, mllib):

        self.param_prefix = "decoder"
        super().__init__(input_dict, param_space, mllib)


    def mirror_encoder(self, encoder, input_dict):
        """Build parameter space by mirroring encoder."""
        
        encoder_layer_types = self.mllib.get_model_layer_type_list(encoder.model_obj)

        # reverse order of layer definitions
        self.layer_params_list = encoder.layer_params_list[::-1].copy()
        self.num_layers = len(self.layer_params_list)

        for decoder_layer_idx, layer_dict in enumerate(self.layer_params_list):
            
            encoder_layer_idx = encoder.num_layers - decoder_layer_idx - 1
            encoder_layer_type = encoder_layer_types[encoder_layer_idx]
            encoder_layer_input_shape, encoder_layer_output_shape = self.mllib.get_layer_io_shape(encoder.model_obj, encoder_layer_idx)

            # convert convolutions to transpose convolutions
            # strides stay the same, but number of filters taken from input shape of original layer
            if encoder_layer_type[:4] == "conv":
                layer_dict["layer_type"] = "trans_conv"
                if input_dict["network_order"] == "NCHW":
                    layer_dict["num_filters"] = encoder_layer_input_shape[0]
                else:
                    layer_dict["num_filters"] = encoder_layer_input_shape[-1]

            # replace flatten with reshape
            elif encoder_layer_type == "flatten":
                layer_dict = {}
                layer_dict["layer_type"] = "reshape"
                layer_dict["target_shape"] = encoder_layer_input_shape
                layer_dict["layer_input_idx"] = -1
                self.layer_params_list[decoder_layer_idx] = layer_dict

            # change output_shape of dense layers to that of the original layer's input
            elif encoder_layer_type == "dense":
                layer_dict["output_size"] = encoder_layer_input_shape[0]

            # ignore input
            elif encoder_layer_type == "input":
                pass

            else:
                raise ValueError("Unexpected encoder layer type during decoder mirroring: " + encoder_layer_type)

            if decoder_layer_idx == (self.num_layers - 1):
                try:
                    final_activation = input_dict["decoder_final_activation"]
                    layer_dict["activation"] = final_activation
                except KeyError:
                    print("Could not find final_activation, using original " + str(layer_dict["activation"]))
                    sleep(1.0)
