import torch
from torchviz import make_dot
from ivadomed.architecture.unet.unet3d import Modified3DUNet
from loguru import logger
from torch.utils.tensorboard import SummaryWriter
from pytorch_model_summary import summary


def test_visualize_3dunet():
    writer = SummaryWriter(f'logs/net')
    x = torch.tensor(
        # Expected 5-dimensional input for 5-dimensional weight [
        # batch size, channel? depth, x, y?
        torch.zeros(5, 1, 16, 48, 48)
    ).float() # This float conversion is VERY necessary
    logger.success(x.shape)

    unet_instance = Modified3DUNet(1, 1, attention=True, base_n_filter=1)

    params = dict(
        list(unet_instance.named_parameters())
    )
    y = unet_instance(x)
    logger.info(y)
    make_dot(y, params=params, show_attrs=True, show_saved=True).render("rnn_torchviz", format="pdf")

    # print(unet_instance)
    # writer.add_graph(unet_instance, inp)
    # summary(unet_instance, (16, 1, 10, 10, 10))
    summary(unet_instance, x, show_input=True, print_summary=True, show_hierarchical=True, show_parent_layers=True)
