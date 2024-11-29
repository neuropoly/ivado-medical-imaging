from ivadomed.architecture.block.encoder_block import EncoderBlock
from ivadomed.architecture.block.decoder_block import DecoderBlock
from unet import Unet


class FiLMedUnet(Unet):
    """U-Net network containing FiLM layers to condition the model with another data type (i.e. not an image).

    Args:
        n_channel (int): Number of channels in the input image.
        out_channel (int): Number of channels in the output image.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        film_layers (list): List of 0 or 1 indicating on which layer FiLM is applied.
        is_2d (bool): Indicates dimensionality of model.
        n_filters (int):  Number of base filters in the U-Net.
        **kwargs:

    Attributes:
        encoder (EncoderBlock): U-Net encoder.
        decoder (DecoderBlock): U-net decoder.
    """

    def __init__(self, in_channel=1, out_channel=1, depth=3, dropout_rate=0.3,
                 bn_momentum=0.1, n_metadata=None, film_layers=None, is_2d=True, n_filters=64, **kwargs):
        super().__init__(in_channel=in_channel, out_channel=out_channel, depth=depth,
                         dropout_rate=dropout_rate, bn_momentum=bn_momentum)

        # Verify if the length of boolean FiLM layers corresponds to the depth
        if film_layers:
            if len(film_layers) != 2 * depth + 2:
                raise ValueError("The number of FiLM layers {} entered does not correspond to the "
                                 "UNet depth. There should 2 * depth + 2 layers.".format(len(film_layers)))
        else:
            film_layers = [0] * (2 * depth + 2)
        # Encoder path
        self.encoder = EncoderBlock(in_channel=in_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)
        # Decoder path
        self.decoder = DecoderBlock(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate, bn_momentum=bn_momentum,
                                    n_metadata=n_metadata, film_layers=film_layers, is_2d=is_2d, n_filters=n_filters)

    def forward(self, x, context=None):
        features, w_film = self.encoder(x, context)
        preds = self.decoder(features, context, w_film)

        return preds