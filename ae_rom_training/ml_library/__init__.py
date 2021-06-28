import os

# Check whether ML libraries are accessible
# Tensorflow-Keras
TFKERAS_IMPORT_SUCCESS = True
try:
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # don't print all the TensorFlow warnings
    import tensorflow as tf
except ImportError:
    TFKERAS_IMPORT_SUCCESS = False

if TFKERAS_IMPORT_SUCCESS:
    from ae_rom_training.ml_library.tfkeras_library import TFKerasLibrary

# PyTorch
TORCH_IMPORT_SUCCESS = True
try:
    import torch
except ImportError:
    TORCH_IMPORT_SUCCESS = False

if TORCH_IMPORT_SUCCESS:
    from ae_rom_training.ml_library.pytorch_library import PyTorchLibrary


def get_ml_library(input_dict):
    """Helper function to retrieve machine learning library helper classes."""

    # check that desired ML library is installed and requested model is compatible
    if input_dict["mllib_name"] == "tfkeras":
        assert TFKERAS_IMPORT_SUCCESS, "Tensorflow failed to import, please check that it is installed"
        mllib = TFKerasLibrary(input_dict)
    elif input_dict["mllib_name"] == "pytorch":
        assert TORCH_IMPORT_SUCCESS, "PyTorch failed to import, please check that it is installed."
        mllib = PyTorchLibrary(input_dict)
    else:
        raise ValueError("Invalid mllib_name: " + str(input_dict["mllib_name"]))

    return mllib