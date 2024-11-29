from torch.nn import Module, Conv2d, BatchNorm2d, LeakyReLU


class ConvBlock(Module):
    def __init__(self, in_chan, out_chan, ksize=3, stride=1, pad=0, activation=LeakyReLU()):
        """
        Perform convolution, activation and batch normalization.
        Args:
            in_chan (int): number of channels on input
            out_chan (int): number of channel on output
            ksize (int): size of kernel for the 2d convolution
            stride (int): strides for 2d convolution
            pad (int): pad for nn.conv2d
            activation (nn.layers): activation layer. default Leaky ReLu
        """
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2d(in_chan, out_chan, kernel_size=ksize, stride=stride, padding=pad)
        self.activation = activation
        self.batch_norm = BatchNorm2d(out_chan)

    def forward(self, x):
        return self.activation(self.batch_norm(self.conv1(x)))
