import torch
from torch.nn import Module, functional, Conv2d, Conv3d, BatchNorm2d, BatchNorm3d, Sequential
from ivadomed.architecture.layers_common import weights_init_kaiming


class GridAttentionBlock(Module):
    """Attention module to focus on important features passed through U-Net's decoder
    Specific to Attention UNet

    .. seealso::
        Oktay, Ozan, et al. "Attention u-net: Learning where to look for the pancreas."
        arXiv preprint arXiv:1804.03999 (2018).

    Args:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.

    Attributes:
        in_channels (int): Number of channels in the input image.
        gating_channels (int): Number of channels in the gating step.
        inter_channels (int): Number of channels in the intermediate gating step.
        dimension (int): Value of 2 or 3 to indicating whether it is used in a 2D or 3D model.
        sub_sample_factor (tuple or list): Convolution kernel size.
        upsample_mode (str): 'bilinear' or 'trilinear' related to the use of 2D or 3D models.
        W (Sequential): Sequence of convolution and batch normalization layers.
        theta (Conv2d or Conv3d): Convolution layer for gating operation.
        phi (Conv2d or Conv3d): Convolution layer for gating operation.
        psi (Conv2d or Conv3d): Convolution layer for gating operation.

    """

    def __init__(self, in_channels, gating_channels, inter_channels=None, dimension=3,
                 sub_sample_factor=(2, 2, 2)):
        super(GridAttentionBlock, self).__init__()

        assert dimension in [2, 3]

        # Downsampling rate for the input featuremap
        if isinstance(sub_sample_factor, tuple):
            self.sub_sample_factor = sub_sample_factor
        elif isinstance(sub_sample_factor, list):
            self.sub_sample_factor = tuple(sub_sample_factor)
        else:
            self.sub_sample_factor = tuple([sub_sample_factor]) * dimension

        # Default parameter set
        self.dimension = dimension
        self.sub_sample_kernel_size = self.sub_sample_factor

        # Number of channels (pixel dimensions)
        self.in_channels = in_channels
        self.gating_channels = gating_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        if dimension == 3:
            conv_nd = Conv3d
            bn = BatchNorm3d
            self.upsample_mode = 'trilinear'
        elif dimension == 2:
            conv_nd = Conv2d
            bn = BatchNorm2d
            self.upsample_mode = 'bilinear'
        else:
            raise NotImplementedError

        # Output transform
        self.W = Sequential(
            conv_nd(in_channels=self.in_channels, out_channels=self.in_channels, kernel_size=1, stride=1, padding=0),
            bn(self.in_channels),
        )

        # Theta^T * x_ij + Phi^T * gating_signal + bias
        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=self.sub_sample_kernel_size, stride=self.sub_sample_factor, padding=0,
                             bias=False)
        self.phi = conv_nd(in_channels=self.gating_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0, bias=True)
        self.psi = conv_nd(in_channels=self.inter_channels, out_channels=1, kernel_size=1, stride=1, padding=0,
                           bias=True)

        # Initialise weights
        for m in self.children():
            m.apply(weights_init_kaiming)

        # Define the operation
        self.operation_function = self._concatenation

    def forward(self, x, g):
        output = self.operation_function(x, g)
        return output

    def _concatenation(self, x, g):
        input_size = x.size()
        batch_size = input_size[0]
        assert batch_size == g.size(0)

        # theta => (b, c, t, h, w) -> (b, i_c, t, h, w) -> (b, i_c, thw)
        # phi   => (b, g_d) -> (b, i_c)
        theta_x = self.theta(x)
        theta_x_size = theta_x.size()

        # g (b, c, t', h', w') -> phi_g (b, i_c, t', h', w')
        #  Relu(theta_x + phi_g + bias) -> f = (b, i_c, thw) -> (b, i_c, t/s1, h/s2, w/s3)
        phi_g = functional.interpolate(self.phi(g), size=theta_x_size[2:], mode=self.upsample_mode, align_corners=True)
        f = functional.relu(theta_x + phi_g, inplace=True)

        #  psi^T * f -> (b, psi_i_c, t/s1, h/s2, w/s3)
        sigm_psi_f = torch.sigmoid(self.psi(f))

        # upsample the attentions and multiply
        sigm_psi_f = functional.interpolate(sigm_psi_f, size=input_size[2:], mode=self.upsample_mode, align_corners=True)
        y = sigm_psi_f.expand_as(x) * x
        W_y = self.W(y)

        return W_y, sigm_psi_f

