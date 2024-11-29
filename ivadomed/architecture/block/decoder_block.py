import torch
from torch.nn import Module, ModuleList, ReLU, Conv2d, Conv3d, Softmax
from ivadomed.architecture.block.up_conv_block import UpConvBlock
from ivadomed.architecture.block.film_layer_block import FiLMlayerBlock


class DecoderBlock(Module):
    """Decoding part of the U-Net model.

    Args:
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        hemis (bool): Boolean indicating if HeMIS is on or not.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        out_channel (int): Number of channels in the output image.
        up_path (ModuleList): List of module operations done during decoding.
        last_conv (Conv2d): Last convolution.
        last_film (FiLMlayerBlock): FiLM layer applied to last convolution.
        softmax (Softmax): Softmax layer that can be applied as last layer.
    """

    def __init__(self, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1,
                 n_metadata=None, film_layers=None, hemis=False, final_activation="sigmoid", is_2d=True, n_filters=64):
        super(DecoderBlock, self).__init__()
        self.depth = depth
        self.out_channel = out_channel
        self.final_activation = final_activation
        # Up-Sampling path
        self.up_path = ModuleList()
        if hemis:
            in_channel = n_filters * 2 ** self.depth
            self.up_path.append(
                UpConvBlock(in_channel * 2, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum,
                            is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayerBlock(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)
            # self.depth += 1
        else:
            in_channel = n_filters * 2 ** self.depth

            self.up_path.append(
                UpConvBlock(in_channel, n_filters * 2 ** (self.depth - 1), dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + 1]:
                self.up_path.append(FiLMlayerBlock(n_metadata, n_filters * 2 ** (self.depth - 1)))
            else:
                self.up_path.append(None)

        for i in range(1, depth):
            in_channel //= 2

            self.up_path.append(
                UpConvBlock(in_channel + n_filters * 2 ** (self.depth - i - 1 + int(hemis)),
                            n_filters * 2 ** (self.depth - i - 1),
                            dropout_rate, bn_momentum, is_2d))
            if film_layers and film_layers[self.depth + i + 1]:
                self.up_path.append(FiLMlayerBlock(n_metadata, n_filters * 2 ** (self.depth - i - 1)))
            else:
                self.up_path.append(None)

        # Last Convolution
        conv = Conv2d if is_2d else Conv3d
        self.last_conv = conv(in_channel // 2, out_channel, kernel_size=3, padding=1)
        self.last_film = FiLMlayerBlock(n_metadata, self.out_channel) if film_layers and film_layers[-1] else None
        self.softmax = Softmax(dim=1)

    def forward(self, features, context=None, w_film=None):
        x = features[-1]

        for i in reversed(range(self.depth)):
            x = self.up_path[-(i + 1) * 2](x, features[i])
            if self.up_path[-(i + 1) * 2 + 1]:
                x, w_film = self.up_path[-(i + 1) * 2 + 1](x, context, w_film)

        # Last convolution
        x = self.last_conv(x)
        if self.last_film:
            x, w_film = self.last_film(x, context, w_film)

        if hasattr(self, "final_activation") and self.final_activation not in ["softmax", "relu", "sigmoid"]:
            raise ValueError("final_activation value has to be either softmax, relu, or sigmoid")
        elif hasattr(self, "final_activation") and self.final_activation == "softmax":
            preds = self.softmax(x)
        elif hasattr(self, "final_activation") and self.final_activation == "relu":
            preds = ReLU()(x) / ReLU()(x).max()
            # If nn.ReLU()(x).max()==0, then nn.ReLU()(x) will also ==0. So, here any zero division will always be 0/0.
            # For float values, 0/0=nan. So, we can handle zero division (without checking data!) by setting nans to 0.
            preds[torch.isnan(preds)] = 0.
            # If model multiclass
            if self.out_channel > 1:
                class_sum = preds.sum(dim=1).unsqueeze(1)
                # Avoid division by zero
                class_sum[class_sum == 0] = 1
                preds /= class_sum
        else:
            preds = torch.sigmoid(x)

        # If model multiclass
        if self.out_channel > 1:
            # Remove background class
            preds = preds[:, 1:, ]

        return preds

