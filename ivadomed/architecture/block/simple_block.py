import torch
from torch.nn import Module, LeakyReLU, MaxPool2d
from ivadomed.architecture.block.conv_block import ConvBlock


class SimpleBlock(Module):
    def __init__(self, in_chan, out_chan_1x1, out_chan_3x3, activation=LeakyReLU()):
        """
        Inception module with 3 convolutions with different kernel size with separate activation.
        The 3 outputs are then concatenated. Max pooling performed on concatenation.
        Args:
            in_chan (int): number of channel of input
            out_chan_1x1 (int): number of channel after first convolution block
            out_chan_3x3 (int): number of channel for the other convolution blocks
            activation (nn.layers): activation layer used in convolution block
        """
        super(SimpleBlock, self).__init__()
        self.conv1 = ConvBlock(in_chan, out_chan_1x1, ksize=3, pad=0, activation=activation)
        self.conv2 = ConvBlock(in_chan, out_chan_3x3, ksize=5, pad=1, activation=activation)
        self.conv3 = ConvBlock(in_chan, out_chan_3x3, ksize=9, pad=3, activation=activation)
        self.MP = MaxPool2d(1)

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(x)
        conv3_out = self.conv3(x)
        output = torch.cat([conv1_out, conv2_out, conv3_out], 1)
        output = self.MP(output)
        return output