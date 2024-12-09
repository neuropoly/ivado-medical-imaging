import pytest
import torch
import torch.backends.cudnn as cudnn
from ivadomed import models as imed_models
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path
from loguru import logger

cudnn.benchmark = True

N_METADATA = 1
OUT_CHANNEL = 1
INITIAL_LR = 0.001


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('fraction', [0.1, 0.2, 0.3])
@pytest.mark.parametrize('path_model', [str(Path(__data_testing_dir__, 'model_unet_test.pt'))])
def test_transfer_learning(download_data_testing_test_files, path_model, fraction, tolerance=0.15):
    device = torch.device("cpu")
    logger.info(f"Working on {'cpu'}.")
    logger.info(__data_testing_dir__)

    # Load pretrained model
    model_pretrained = torch.load(path_model, map_location=device)
    # Setup model for retrain
    model_to_retrain = imed_models.set_model_for_retrain(path_model, retrain_fraction=fraction,
                                                         map_location=device)

    logger.info(f"\nSet fraction to retrain: {fraction}")

    # Check Frozen part
    grad_list = [param.requires_grad for name, param in model_to_retrain.named_parameters()]
    fraction_retrain_measured = sum(grad_list) * 1.0 / len(grad_list)
    logger.debug(f"\nMeasure: retrained fraction of the model: {round(fraction_retrain_measured, 1)}")
    # for name, param in model.named_parameters():
    #    print("\t", name, param.requires_grad)
    assert (abs(fraction_retrain_measured - fraction) <= tolerance)
    total_params = sum(p.numel() for p in model_to_retrain.parameters())
    logger.info(f"{total_params} total parameters.")
    total_trainable_params = sum(
        p.numel() for p in model_to_retrain.parameters() if p.requires_grad)
    logger.info(f"{total_trainable_params} parameters to retrain.")
    assert (total_params > total_trainable_params)

    # Check reset weights
    reset_list = [(p1.data.ne(p2.data).sum() > 0).cpu().numpy()
                  for p1, p2 in zip(model_pretrained.parameters(), model_to_retrain.parameters())]
    reset_measured = sum(reset_list) * 1.0 / len(reset_list)
    logger.info(f"\nMeasure: reset fraction of the model: {round(reset_measured, 1)}")
    assert (abs(reset_measured - fraction) <= tolerance)
    # weights_reset = False
    # for name_p1, p2 in zip(model_copy.named_parameters(), model.parameters()):
    #    if name_p1[1].data.ne(p2.data).sum() > 0:
    #        print('\t', name_p1[0], True)
    #        weights_reset = True
    #    else:
    #        print('\t', name_p1[0], False)
    # assert(weights_reset)


def teardown_function():
    remove_tmp_dir()
