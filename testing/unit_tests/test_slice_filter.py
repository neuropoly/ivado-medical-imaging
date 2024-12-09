import pytest
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from loguru import logger

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir

cudnn.benchmark = True

GPU_ID = 0
BATCH_SIZE = 1


def setup_function():
    create_tmp_dir()


def _cmpt_slice(ds_loader):
    cmpt_label = {0: 0, 1: 0}
    for i, batch in enumerate(ds_loader):
        for gt in batch['gt']:
            # TODO: multi label
            if np.any(gt.numpy()):
                cmpt_label[1] += 1
            else:
                cmpt_label[0] += 1
    logger.debug(cmpt_label)
    return cmpt_label[0], cmpt_label[1]


@pytest.mark.parametrize('transforms_dict', [
    {"Resample": {"wspace": 0.75, "hspace": 0.75},
     "ROICrop": {"size": [48, 48]},
     "NumpyToTensor": {}},
    {"Resample": {"wspace": 0.75, "hspace": 0.75, "applied_to": ["im", "gt"]},
     "CenterCrop": {"size": [100, 100], "applied_to": ["im", "gt"]},
     "NumpyToTensor": {"applied_to": ["im", "gt"]}}])
@pytest.mark.parametrize('train_lst', [['sub-unf01_T2w.nii.gz']])
@pytest.mark.parametrize('target_lst', [["_lesion-manual"]])
@pytest.mark.parametrize('slice_filter_params', [
    {"filter_empty_mask": False, "filter_empty_input": True},
    {"filter_empty_mask": True, "filter_empty_input": True}])
@pytest.mark.parametrize('roi_params', [
    {"suffix": "_seg-manual", "slice_filter_roi": 10},
    {"suffix": None, "slice_filter_roi": 0}])
def test_slice_filter(download_data_testing_test_files, transforms_dict, train_lst, target_lst, roi_params, slice_filter_params):
    if "ROICrop" in transforms_dict and roi_params["suffix"] is None:
        return

    cuda_available, device = imed_utils.define_device(GPU_ID)

    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": train_lst,
        "dataset_type": "training",
        "requires_undo": False,
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "path_data": [__data_testing_dir__],
        "target_suffix": target_lst,
        "extensions": [".nii.gz"],
        "roi_params": roi_params,
        "model_params": {"name": "Unet"},
        "slice_filter_params": slice_filter_params,
        "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "slice_axis": "axial",
        "multichannel": False
    }
    # Get Training dataset
    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)
    ds_train = imed_loader.load_dataset(bids_df, **loader_params)

    logger.info(f"\tNumber of loaded slices: {len(ds_train)}")

    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)
    cmpt_neg, cmpt_pos = _cmpt_slice(train_loader)
    if slice_filter_params["filter_empty_mask"]:
        assert cmpt_neg == 0
        assert cmpt_pos != 0
    else:
        # We verify if there are still some negative slices (they are removed with our filter)
        assert cmpt_neg != 0 and cmpt_pos != 0
    logger.info(f"\tNumber of Neg/Pos slices in GT: {cmpt_neg/cmpt_pos}")


def teardown_function():
    remove_tmp_dir()
