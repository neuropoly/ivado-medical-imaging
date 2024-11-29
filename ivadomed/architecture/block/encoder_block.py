from torch import nn
from torch.nn import Module
from ivadomed.architecture.block.down_conv_block import DownConvBlock
from ivadomed.architecture.block.film_layer_block import FiLMlayerBlock


class EncoderBlock(Module):
    """Encoding part of the U-Net model.
    It returns the features map for the skip connections

    Args:
        in_channel (int): Number of channels in the input image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        down_path (ModuleList): List of module operations done during encoding.
        conv_bottom (DownConvBlock): Bottom down convolution.
        film_bottom (FiLMlayerBlock): FiLM layer applied to bottom convolution.
    """

    def __init__(self, in_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, n_metadata=None, film_layers=None,
                 is_2d=True, n_filters=64):
        super(EncoderBlock, self).__init__()
        self.depth = depth
        self.down_path = nn.ModuleList()
        # first block
        self.down_path.append(DownConvBlock(in_channel, n_filters, dropout_rate, bn_momentum, is_2d))
        self.down_path.append(FiLMlayerBlock(n_metadata, n_filters) if film_layers and film_layers[0] else None)
        max_pool = nn.MaxPool2d if is_2d else nn.MaxPool3d
        self.down_path.append(max_pool(2))

        for i in range(depth - 1):
            self.down_path.append(DownConvBlock(n_filters, n_filters * 2, dropout_rate, bn_momentum, is_2d))
            self.down_path.append(FiLMlayerBlock(n_metadata, n_filters * 2) if film_layers and film_layers[i + 1] else None)
            self.down_path.append(max_pool(2))
            n_filters = n_filters * 2

        # Bottom
        self.conv_bottom = DownConvBlock(n_filters, n_filters, dropout_rate, bn_momentum, is_2d)
        self.film_bottom = FiLMlayerBlock(n_metadata, n_filters) if film_layers and film_layers[self.depth] else None

    def forward(self, x, context=None):
        features = []

        # First block
        x = self.down_path[0](x)
        if self.down_path[1]:
            x, w_film = self.down_path[1](x, context, None)
        features.append(x)
        x = self.down_path[2](x)

        # Down-sampling path (other blocks)
        for i in range(1, self.depth):
            x = self.down_path[i * 3](x)
            if self.down_path[i * 3 + 1]:
                x, w_film = self.down_path[i * 3 + 1](x, context, None if 'w_film' not in locals() else w_film)
            features.append(x)
            x = self.down_path[i * 3 + 2](x)

        # Bottom level
        x = self.conv_bottom(x)
        if self.film_bottom:
            x, w_film = self.film_bottom(x, context, None if 'w_film' not in locals() else w_film)
        features.append(x)

        return features, None if 'w_film' not in locals() else w_film

