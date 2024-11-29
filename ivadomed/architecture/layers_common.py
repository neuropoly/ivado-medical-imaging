from pathlib import Path

import torch
from torch.nn import init


def weights_init_kaiming(m):
    """Initialize weights according to method describe here:
    https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/He_Delving_Deep_into_ICCV_2015_paper.pdf
    """

    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)


def set_model_for_retrain(model_path, retrain_fraction, map_location, reset=True):
    """Set model for transfer learning.

    The first layers (defined by 1-retrain_fraction) are frozen (i.e. requires_grad=False).
    The weights of the last layers (defined by retrain_fraction) are reset unless reset option is False.

    Args:
        model_path (str): Pretrained model path.
        retrain_fraction (float): Fraction of the model that will be retrained, between 0 and 1. If set to 0.3,
            then the 30% last fraction of the model will be re-initalised and retrained.
        map_location (str): Device.
        reset (bool): if the un-frozen weight should be reset or kept as loaded.

    Returns:
        torch.Module: Model ready for retrain.
    """
    # Load pretrained model
    model = torch.load(model_path, map_location=map_location)
    # Get number of layers with learnt parameters
    layer_names = [name for name, layer in model.named_modules() if hasattr(layer, 'reset_parameters')]
    n_layers = len(layer_names)
    # Compute the number of these layers we want to freeze
    n_freeze = int(round(n_layers * (1 - retrain_fraction)))
    # Last frozen layer
    last_frozen_layer = layer_names[n_freeze]

    # Set freeze first layers
    for name, layer in model.named_parameters():
        if not name.startswith(last_frozen_layer):
            layer.requires_grad = False
        else:
            break

    # Reset weights of the last layers
    if reset:
        for name, layer in model.named_modules():
            if name in layer_names[n_freeze:]:
                layer.reset_parameters()

    return model


def get_model_filenames(folder_model):
    """Get trained model filenames from its folder path.

    This function checks if the folder_model exists and get trained model (.pt or .onnx) and its configuration file
    (.json) from it.
    Note: if the model exists as .onnx, then this function returns its onnx path instead of the .pt version.

    Args:
        folder_name (str): Path of the model folder.

    Returns:
        str, str: Paths of the model (.onnx) and its configuration file (.json).
    """
    if Path(folder_model).is_dir():
        prefix_model = Path(folder_model).name
        # Check if model and model metadata exist. Verify if ONNX model exists, if not try to find .pt model
        fname_model = Path(folder_model, prefix_model + '.onnx')
        if not fname_model.is_file():
            fname_model = Path(folder_model, prefix_model + '.pt')
            if not fname_model.exists():
                raise FileNotFoundError(fname_model)
        fname_model_metadata = Path(folder_model, prefix_model + '.json')
        if not fname_model_metadata.is_file():
            raise FileNotFoundError(fname_model)
    else:
        raise FileNotFoundError(folder_model)
    return str(fname_model), str(fname_model_metadata)
