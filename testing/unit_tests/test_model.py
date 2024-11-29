import ivadomed.architecture.block.film_generator
import ivadomed.architecture.densenet
import ivadomed.architecture.resnet
import ivadomed.architecture.unet
import ivadomed.architecture.unet.unet3d
import ivadomed.models as imed_model
import torch
import torchvision


def test_countception():
    a = [[[[0 for i in range(10)] for i in range(10)]]]
    inp = torch.tensor(a).float()
    model = imed_model.Countception(in_channel=1, out_channel=1)
    inf = model(inp)
    assert (type(inf) == torch.Tensor)


def test_model_3d_att():
    # verifying if 3d attention model can be created
    a = [[[[[0 for i in range(48)] for j in range(48)] for k in range(16)]]]
    inp = torch.tensor(a).float()
    model = ivadomed.architecture.unet.unet3d.Modified3DUNet(in_channel=1, out_channel=1, attention=True)
    inf = model(inp)
    assert(type(inf) == torch.Tensor)


def test_resnet():
    a = [[[[0 for i in range(100)] for i in range(100)]]]
    inp = torch.tensor(a).float()
    model = ivadomed.architecture.resnet.ResNet(torchvision.models.resnet.BasicBlock, [2, 2, 2, 2])
    inf = model(inp)
    assert (type(inf) == torch.Tensor)


def test_densenet():
    a = [[[[0 for i in range(100)] for i in range(100)]]]
    inp = torch.tensor(a).float()
    model = ivadomed.architecture.densenet.DenseNet(32, (6, 12, 24, 16), 64)
    inf = model(inp)
    assert (type(inf) == torch.Tensor)


def test_filmed_unet():
    a = [[[[0 for i in range(100)] for i in range(100)]]]
    inp = torch.tensor(a).float()
    model = ivadomed.architecture.unet.unet.FiLMedUnet()
    inf = model(inp)
    assert (type(inf) == torch.Tensor)


def test_film_generator():
    a = [[[[0 for i in range(64)] for i in range(64)]]]
    inp = torch.tensor(a).float()
    model = ivadomed.architecture.block.film_generator.FiLMgenerator(64, 1)
    inf = model(inp)
    assert (type(inf[0]) == torch.Tensor)
    assert (type(inf[1]) == list)
    assert (len(inf[1]) == 2)
    assert (type(inf[1][0]) == torch.nn.parameter.Parameter)
    assert (type(inf[1][1]) == torch.nn.parameter.Parameter)
