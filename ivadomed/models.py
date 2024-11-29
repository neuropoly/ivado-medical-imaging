import torch
import torch.nn as nn
from torch.nn import Module
from torch.nn import init
from pathlib import Path

#Modified from torchvision.models.resnet.Resnet
from ivadomed.architecture.block.conv_block import ConvBlock
from ivadomed.architecture.block.simple_block import SimpleBlock


class Countception(Module):
    """Countception model.
    Fully convolutional model using inception module and used for key points detection.
    The inception model extracts several patches within each image. Every pixel is therefore processed by the
    network several times, allowing to average multiple predictions and minimize false negatives.

    .. seealso::
        Cohen JP et al. "Count-ception: Counting by fully convolutional redundant counting."
        Proceedings of the IEEE International Conference on Computer Vision Workshops. 2017.

    Args:
        in_channel (int): number of channels on input image
        out_channel (int): number of channels on output image
        use_logits (bool): boolean to change output
        logits_per_output (int): number of outputs of final convolution which will multiplied by the number of channels
        name (str): model's name used for call in configuration file.
    """

    def __init__(self, in_channel=3, out_channel=1, use_logits=False, logits_per_output=12, name='CC', **kwargs):
        super(Countception, self).__init__()

        # params
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.activation = nn.LeakyReLU(0.01)
        self.final_activation = nn.LeakyReLU(0.3)
        self.patch_size = 40
        self.use_logits = use_logits
        self.logits_per_output = logits_per_output

        torch.LongTensor()

        self.conv1 = ConvBlock(self.in_channel, 64, ksize=3, pad=self.patch_size, activation=self.activation)
        self.simple1 = SimpleBlock(64, 16, 16, activation=self.activation)
        self.simple2 = SimpleBlock(48, 16, 32, activation=self.activation)
        self.conv2 = ConvBlock(80, 16, ksize=14, activation=self.activation)
        self.simple3 = SimpleBlock(16, 112, 48, activation=self.activation)
        self.simple4 = SimpleBlock(208, 64, 32, activation=self.activation)
        self.simple5 = SimpleBlock(128, 40, 40, activation=self.activation)
        self.simple6 = SimpleBlock(120, 32, 96, activation=self.activation)
        self.conv3 = ConvBlock(224, 32, ksize=20, activation=self.activation)
        self.conv4 = ConvBlock(32, 64, ksize=10, activation=self.activation)
        self.conv5 = ConvBlock(64, 32, ksize=9, activation=self.activation)

        if use_logits:
            self.conv6 = nn.ModuleList([ConvBlock(
                64, logits_per_output, ksize=1, activation=self.final_activation) for _ in range(out_channel)])
        else:
            self.conv6 = ConvBlock(32, self.out_channel, ksize=20, pad=1, activation=self.final_activation)

        # Weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                init.xavier_uniform_(m.weight, gain=init.calculate_gain('leaky_relu', param=0.01))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        net = self.conv1(x)  # 32
        net = self.simple1(net)
        net = self.simple2(net)
        net = self.conv2(net)
        net = self.simple3(net)
        net = self.simple4(net)
        net = self.simple5(net)
        net = self.simple6(net)
        net = self.conv3(net)
        net = self.conv4(net)
        net = self.conv5(net)

        if self.use_logits:
            net = [c(net) for c in self.conv6]
            [self._print(n) for n in net]
        else:
            net = self.conv6(net)

        return net


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

    This function checks if the folder_model exists and get trained model path (.pt or .onnx based on
    model and GPU availability) and its configuration file (.json) from it.

    Args:
        folder_name (str): Path of the model folder.

    Returns:
        str, str: Paths of the model (.pt or .onnx) and its configuration file (.json).
    """
    if Path(folder_model).is_dir():
        prefix_model = Path(folder_model).name
        fname_model_onnx = Path(folder_model, prefix_model + '.onnx')
        fname_model_pt = Path(folder_model, prefix_model + '.pt')
        cuda_available = torch.cuda.is_available()

        # Assign '.pt' or '.onnx' model based on file existence and GPU/CPU device availability
        if not fname_model_pt.is_file() and not fname_model_onnx.is_file():
            raise FileNotFoundError(f"Model files not found in model folder: "
                                    f"'{str(fname_model_onnx)}' or '{str(fname_model_pt)}'")
        # '.pt' is preferred on GPU, or on CPU if '.onnx' doesn't exist
        elif ((    cuda_available and     fname_model_pt.is_file()) or
              (not cuda_available and not fname_model_onnx.is_file())):
            fname_model = fname_model_pt
        # '.onnx' is preferred on CPU, or on GPU if '.pt' doesn't exist
        elif ((not cuda_available and     fname_model_onnx.is_file()) or
              (    cuda_available and not fname_model_pt.is_file())):
            fname_model = fname_model_onnx

        fname_model_metadata = Path(folder_model, prefix_model + '.json')
        if not fname_model_metadata.is_file():
            raise FileNotFoundError(f"Model config file not found in model folder: '{str(fname_model_metadata)}'")
    else:
        raise FileNotFoundError(folder_model)

    return str(fname_model), str(fname_model_metadata)

