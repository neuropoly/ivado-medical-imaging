import torch
from torch.nn import Module, functional
from ivadomed.architecture.block.down_conv_block import DownConvBlock


class UpConvBlock(Module):
    """2D down convolution.
    Used in U-Net's decoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        downconv (DownConvBlock): Down convolution.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(UpConvBlock, self).__init__()
        self.is_2d = is_2d
        self.downconv = DownConvBlock(in_feat, out_feat, dropout_rate, bn_momentum, is_2d)

    def forward(self, x, y):
        # For retrocompatibility purposes
        if not hasattr(self, "is_2d"):
            self.is_2d = True
        mode = 'bilinear' if self.is_2d else 'trilinear'
        dims = -2 if self.is_2d else -3
        x = functional.interpolate(x, size=y.size()[dims:], mode=mode, align_corners=True)
        x = torch.cat([x, y], dim=1)
        x = self.downconv(x)
        return x
