from torch.nn import Module
from ivadomed.architecture.block.encoder_block import EncoderBlock
from ivadomed.architecture.block.decoder_block import DecoderBlock


class Unet(Module):
    """A reference U-Net model.

    .. seealso::
        Ronneberger, O., et al (2015). U-Net: Convolutional
        Networks for Biomedical Image Segmentation
        ArXiv link: https://arxiv.org/abs/1505.04597

    Args:
        in_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        final_activation (str): Choice of final activation between "sigmoid", "relu" and "softmax".
        is_2d (bool): Indicates dimensionality of model: True for 2D convolutions, False for 3D convolutions.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (ivadomed.architecture.blocks.EncoderBlock): U-Net encoder.
        decoder (ivadomed.architecture.blocks.DecoderBlock): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, final_activation='sigmoid',
                 is_2d=True, n_filters=64, **kwargs):
        super(Unet, self).__init__()

        # Encoder path
        self.encoder = EncoderBlock(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    is_2d=is_2d, n_filters=n_filters)

        # Decoder path
        self.decoder = DecoderBlock(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    final_activation=final_activation, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x):
        features, _ = self.encoder(x)
        preds = self.decoder(features)

        return preds


