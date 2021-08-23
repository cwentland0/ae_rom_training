import os
import GPUtil

def get_ml_library(mllib_name, run_gpu):
    """Helper function to retrieve machine learning library helper classes."""

    # set CUDA_VISIBLE_DEVICES to the GPU which has the lowest mem and util
    if run_gpu:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        DEVICE_ID_LIST = GPUtil.getFirstAvailable(order="memory", maxLoad=0.25, maxMemory=0.25)
        DEVICE_ID = DEVICE_ID_LIST[0]  # grab first element from list
        os.environ["CUDA_VISIBLE_DEVICES"] = str(DEVICE_ID)

    # Check whether ML libraries are accessible
    # Tensorflow-Keras
    TFKERAS_IMPORT_SUCCESS = True
    try:
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # don't print all the TensorFlow warnings
        import tensorflow as tf
    except ImportError:
        TFKERAS_IMPORT_SUCCESS = False

    if TFKERAS_IMPORT_SUCCESS:
        from ae_rom_training.ml_library.tfkeras.tfkeras_library import TFKerasLibrary

    # PyTorch
    TORCH_IMPORT_SUCCESS = True
    try:
        import torch
    except ImportError:
        TORCH_IMPORT_SUCCESS = False

    if TORCH_IMPORT_SUCCESS:
        from ae_rom_training.ml_library.pytorch.pytorch_library import PyTorchLibrary

    # check that desired ML library is installed and requested model is compatible
    if mllib_name == "tfkeras":
        assert TFKERAS_IMPORT_SUCCESS, "Tensorflow failed to import, please check that it is installed"
        mllib = TFKerasLibrary(run_gpu)
    elif mllib_name == "pytorch":
        assert TORCH_IMPORT_SUCCESS, "PyTorch failed to import, please check that it is installed."
        mllib = PyTorchLibrary(run_gpu)
    else:
        raise ValueError("Invalid mllib_name: " + str(mllib_name))

    return mllib
