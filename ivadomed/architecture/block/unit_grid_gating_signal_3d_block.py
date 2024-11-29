from torch.nn import Module,  Sequential, Conv3d, BatchNorm3d, ReLU
from ivadomed.architecture.layers_common import weights_init_kaiming


class UnetGridGatingSignal3DBlock(Module):
    """Operation to extract important features for a specific task using 1x1x1 convolution (Gating) which is used in the
    attention blocks.

    Args:
        in_size (int): Number of channels in the input image.
        out_size (int): Number of channels in the output image.
        kernel_size (tuple): Convolution kernel size.
        is_batchnorm (bool): Boolean indicating whether to apply batch normalization or not.

    Attributes:
        conv1 (Sequential): 3D convolution, batch normalization and ReLU activation.
    """

    def __init__(self, in_size, out_size, kernel_size=(1, 1, 1), is_batchnorm=True):
        super(UnetGridGatingSignal3DBlock, self).__init__()

        if is_batchnorm:
            self.conv1 = Sequential(Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       BatchNorm3d(out_size),
                                       ReLU(inplace=True),
                                       )
        else:
            self.conv1 = Sequential(Conv3d(in_size, out_size, kernel_size, (1, 1, 1), (0, 0, 0)),
                                       ReLU(inplace=True),
                                       )

        # initialise the blocks
        for m in self.children():
            weights_init_kaiming(m)

    def forward(self, inputs):
        outputs = self.conv1(inputs)
        return outputs

