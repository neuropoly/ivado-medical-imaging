import torch
from torch.nn import Module, ModuleDict
from ivadomed.architecture.block.encoder_block import EncoderBlock
from ivadomed.architecture.block.decoder_block import DecoderBlock


class HeMISUnet(Module):
    """A U-Net model inspired by HeMIS to deal with missing contrasts.
        1) It has as many encoders as contrasts but only one decoder.
        2) Skip connections are the concatenations of the means and var of all encoders skip connections.

        Param:
        contrasts: list of all the possible contrasts. ['T1', 'T2', 'T2S', 'F']

    .. seealso::
        Havaei, M., Guizard, N., Chapados, N., Bengio, Y.:
        Hemis: Hetero-modal image segmentation.
        ArXiv link: https://arxiv.org/abs/1607.05194

        Reuben Dorent and Samuel Joutard and Marc Modat and Sébastien Ourselin and Tom Vercauteren
        Hetero-Modal Variational Encoder-Decoder for Joint Modality Completion and Segmentation
        ArXiv link: https://arxiv.org/abs/1907.11150

    Args:
        contrasts (list): List of contrasts.
        out_channel (int): Number of output channels.
        depth (int): Number of down convolutions minus bottom down convolution.
        dropout_rate (float): Probability of dropout.
        bn_momentum (float): Batch normalization momentum.
        **kwargs:

    Attributes:
        depth (int): Number of down convolutions minus bottom down convolution.
        contrasts (list): List of contrasts.
        Encoder_mod (ModuleDict): Contains encoder for each modality.
        decoder (DecoderBlock): U-Net decoder.
    """

    def __init__(self, contrasts, out_channel=1, depth=3, dropout_rate=0.3, bn_momentum=0.1, **kwargs):
        super(HeMISUnet, self).__init__()
        self.depth = depth
        self.contrasts = contrasts

        # Encoder path
        self.Encoder_mod = ModuleDict(
            [['Encoder_{}'.format(Mod), EncoderBlock(in_channel=1, depth=depth, dropout_rate=dropout_rate,
                                                     bn_momentum=bn_momentum)] for Mod in self.contrasts])

        # Decoder path
        self.decoder = DecoderBlock(out_channel=out_channel, depth=depth, dropout_rate=dropout_rate,
                                    bn_momentum=bn_momentum, hemis=True)

    def forward(self, x_mods, indexes_mod):
        """
        X is list like X = [x_T1, x_T2, x_T2S, x_F]
        indexes_mod: list of arrays like [[1, 1, 1], [1, 1, 0], [1, 0, 1], [1, 1, 0]]
        N.B. len(list) = number of contrasts.
        len(list[i]) = Batch size
        """
        features_mod = [[] for _ in range(self.depth + 1)]

        # Down-sampling
        for i, Mod in enumerate(self.contrasts):
            features, _ = self.Encoder_mod['Encoder_{}'.format(Mod)](x_mods[i])

            for j in range(self.depth + 1):
                features_mod[j].append(features[j].unsqueeze(0))

        # Abstraction
        for j in range(self.depth + 1):
            features_cat = torch.cat(features_mod[j], 0).transpose(0, 1)

            features_mod[j] = torch.cat([torch.cat([features_cat[i][indexes_mod[i]].squeeze(1).mean(0),
                                                    features_cat[i][indexes_mod[i]].squeeze(1).var(0)], 0).unsqueeze(0)
                                         for i in range(len(indexes_mod))], 0)

        # Up-sampling
        preds = self.decoder(features_mod)

        return preds
