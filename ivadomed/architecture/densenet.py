from collections import OrderedDict

import torch
import torchvision.models
from torch.nn import functional, Module, Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d, Conv2d, Linear, init, BatchNorm2d


class DenseNet(Module):
    """Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        dropout_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"article" <https://arxiv.org/pdf/1707.06990.pdf>`_
    """

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, dropout_rate=0.3, num_classes=2, memory_efficient=False):

        super(DenseNet, self).__init__()

        # First convolution
        self.features = Sequential(OrderedDict([
            ('conv0', Conv2d(1, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', BatchNorm2d(num_init_features)),
            ('relu0', ReLU(inplace=True)),
            ('pool0', MaxPool2d(kernel_size=3, stride=2, padding=1)),
        ]))

        # Each denseblock
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = torchvision.models.densenet._DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=dropout_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = torchvision.models.densenet._Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', BatchNorm2d(num_features))

        # Linear layer
        self.classifier = Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, Linear):
                init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = functional.relu(features, inplace=True)
        out = functional.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        preds = functional.softmax(out, dim=1)
        # Remove background class
        preds = preds[:, 1:]
        return preds


class densenet121(DenseNet):
    def __init__(self, **kwargs):
        super().__init__(32, (6, 12, 24, 16), 64)