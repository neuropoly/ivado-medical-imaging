import torch
from torch.nn import Module
from ivadomed.architecture.block.film_generator import FiLMgenerator


class FiLMlayerBlock(Module):
    """Applies Feature-wise Linear Modulation to the incoming data
    .. seealso::
        Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer."
        Thirty-Second AAAI Conference on Artificial Intelligence. 2018.

    Args:
        n_metadata (dict): FiLM metadata see ivadomed.loader.film for more details.
        n_channels (int): Number of output channels.

    Attributes:
        batch_size (int): Batch size.
        height (int): Image height.
        width (int): Image width.
        feature_size (int): Number of features in data.
        generator (FiLMgenerator): FiLM network.
        gammas (float): Multiplicative term of the FiLM linear modulation.
        betas (float): Additive term of the FiLM linear modulation.
    """

    def __init__(self, n_metadata, n_channels):
        super(FiLMlayerBlock, self).__init__()

        self.batch_size = None
        self.height = None
        self.width = None
        self.depth = None
        self.feature_size = None
        self.generator = FiLMgenerator(n_metadata, n_channels)
        # Add the parameters gammas and betas to access them out of the class.
        self.gammas = None
        self.betas = None

    def forward(self, feature_maps, context, w_shared):
        data_shape = feature_maps.data.shape
        if len(data_shape) == 4:
            _, self.feature_size, self.height, self.width = data_shape
        elif len(data_shape) == 5:
            _, self.feature_size, self.height, self.width, self.depth = data_shape
        else:
            raise ValueError("Data should be either 2D (tensor length: 4) or 3D (tensor length: 5), found shape: {}".format(data_shape))

        if torch.cuda.is_available():
            context = torch.Tensor(context).cuda()
        else:
            context = torch.Tensor(context)

        # Estimate the FiLM parameters using a FiLM generator from the contioning metadata
        film_params, new_w_shared = self.generator(context, w_shared)

        # FiLM applies a different affine transformation to each channel,
        # consistent accross spatial locations
        if len(data_shape) == 4:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width)
        else:
            film_params = film_params.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            film_params = film_params.repeat(1, 1, self.height, self.width, self.depth)

        self.gammas = film_params[:, :self.feature_size, ]
        self.betas = film_params[:, self.feature_size:, ]

        # Apply the linear modulation
        output = self.gammas * feature_maps + self.betas

        return output, new_w_shared

