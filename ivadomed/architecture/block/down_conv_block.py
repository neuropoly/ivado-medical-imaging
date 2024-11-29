from torch.nn import Module, Conv2d, BatchNorm2d, Dropout2d, Conv3d, Dropout3d, InstanceNorm3d, functional


class DownConvBlock(Module):
    """Two successive series of down convolution, batch normalization and dropout in 2D.
    Used in U-Net's encoder.

    Args:
        in_feat (int): Number of channels in the input image.
        out_feat (int): Number of channels in the output image.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.

    Attributes:
        conv1 (Conv2d): First 2D down convolution with kernel size 3 and padding of 1.
        conv1_bn (BatchNorm2d): First 2D batch normalization.
        conv1_drop (Dropout2d): First 2D dropout.
        conv2 (Conv2d): Second 2D down convolution with kernel size 3 and padding of 1.
        conv2_bn (BatchNorm2d): Second 2D batch normalization.
        conv2_drop (Dropout2d): Second 2D dropout.
    """

    def __init__(self, in_feat, out_feat, dropout_rate=0.3, bn_momentum=0.1, is_2d=True):
        super(DownConvBlock, self).__init__()
        if is_2d:
            conv = Conv2d
            bn = BatchNorm2d
            dropout = Dropout2d
        else:
            conv = Conv3d
            bn = InstanceNorm3d
            dropout = Dropout3d

        self.conv1 = conv(in_feat, out_feat, kernel_size=3, padding=1)
        self.conv1_bn = bn(out_feat, momentum=bn_momentum)
        self.conv1_drop = dropout(dropout_rate)

        self.conv2 = conv(out_feat, out_feat, kernel_size=3, padding=1)
        self.conv2_bn = bn(out_feat, momentum=bn_momentum)
        self.conv2_drop = dropout(dropout_rate)

    def forward(self, x):
        x = functional.relu(self.conv1(x))
        x = self.conv1_bn(x)
        x = self.conv1_drop(x)

        x = functional.relu(self.conv2(x))
        x = self.conv2_bn(x)
        x = self.conv2_drop(x)
        return x
