from torch.nn import Module, Linear, Sigmoid


class FiLMgenerator(Module):
    """The FiLM generator processes the conditioning information
    and produces parameters that describe how the target network should alter its computation.

    Here, the FiLM generator is a multi-layer perceptron.

    Args:
        n_features (int): Number of input channels.
        n_channels (int): Number of output channels.
        n_hid (int): Number of hidden units in layer.

    Attributes:
        linear1 (Linear): Input linear layer.
        sig (Sigmoid): Sigmoid function.
        linear2 (Linear): Hidden linear layer.
        linear3 (Linear): Output linear layer.
    """

    def __init__(self, n_features, n_channels, n_hid=64):
        super(FiLMgenerator, self).__init__()
        self.linear1 = Linear(n_features, n_hid)
        self.sig = Sigmoid()
        self.linear2 = Linear(n_hid, n_hid // 4)
        self.linear3 = Linear(n_hid // 4, n_channels * 2)

    def forward(self, x, shared_weights=None):
        if shared_weights is not None:  # weight sharing
            self.linear1.weight = shared_weights[0]
            self.linear2.weight = shared_weights[1]

        x = self.linear1(x)
        x = self.sig(x)
        x = self.linear2(x)
        x = self.sig(x)
        x = self.linear3(x)

        out = self.sig(x)
        return out, [self.linear1.weight, self.linear2.weight]