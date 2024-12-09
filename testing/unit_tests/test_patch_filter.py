import os
import pytest
import numpy as np
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from loguru import logger

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__, download_data_testing_test_files, path_repo_root
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


@pytest.mark.parametrize('transforms_dict', [{"CenterCrop": {"size": [128, 128], "applied_to": ["im", "gt"]}}])
@pytest.mark.parametrize('train_lst', [['sub-rat3_ses-01_sample-data9_SEM.png']])
@pytest.mark.parametrize('target_lst', [["_seg-axon-manual", "_seg-myelin-manual"]])
@pytest.mark.parametrize('patch_filter_params', [
    {"filter_empty_mask": False, "filter_empty_input":  True},
    {"filter_empty_mask": True, "filter_empty_input": True}])
@pytest.mark.parametrize('dataset_type', ["training", "testing"])
def test_patch_filter_2d(download_data_testing_test_files, transforms_dict, train_lst, target_lst, patch_filter_params,
    dataset_type):

    cuda_available, device = imed_utils.define_device(GPU_ID)

    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": train_lst,
        "dataset_type": dataset_type,
        "requires_undo": False,
        "contrast_params": {"contrast_lst": ['SEM'], "balance": {}},
        "path_data": [os.path.join(__data_testing_dir__, "microscopy_png")],
        "target_suffix": target_lst,
        "extensions": [".png"],
        "roi_params": {"suffix": None, "slice_filter_roi": None},
        "model_params": {"name": "Unet", "length_2D": [32, 32], "stride_2D": [32, 32]},
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "patch_filter_params": patch_filter_params,
        "slice_axis": "axial",
        "multichannel": False
    }
    # Get Training dataset
    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)
    ds = imed_loader.load_dataset(bids_df, **loader_params)

    logger.info(f"\tNumber of loaded patches: {len(ds)}")

    loader = DataLoader(ds, batch_size=BATCH_SIZE,
                        shuffle=True, pin_memory=True,
                        collate_fn=imed_loader_utils.imed_collate,
                        num_workers=0)
    logger.info("\tNumber of Neg/Pos patches in GT.")
    cmpt_neg, cmpt_pos = _cmpt_slice(loader)
    if patch_filter_params["filter_empty_mask"]:
        if dataset_type == "testing":
            # Filters on patches are not applied at testing time
            assert cmpt_neg + cmpt_pos == len(ds)
        else:
            # Filters on patches are applied at training time
            assert cmpt_neg == 0
            assert cmpt_pos != 0
    else:
        # We verify if there are still some negative patches (they are removed with our filter)
        assert cmpt_neg != 0 and cmpt_pos != 0

@pytest.mark.parametrize('transforms_dict', [{"CenterCrop": {"size": [128, 128, 128], "applied_to": ["im", "gt"]}}])
@pytest.mark.parametrize('train_lst', [['sub-unf01_T2w.nii.gz']])
@pytest.mark.parametrize('target_lst', [["_seg-manual"]])
@pytest.mark.parametrize('patch_filter_params', [
    {"filter_empty_mask": False, "filter_empty_input":  True},
    {"filter_empty_mask": True, "filter_empty_input": True}])
@pytest.mark.parametrize('dataset_type', ["training", "testing"])
def test_patch_filter_3d(download_data_testing_test_files, transforms_dict, train_lst, target_lst, patch_filter_params,
    dataset_type):

    cuda_available, device = imed_utils.define_device(GPU_ID)

    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": train_lst,
        "dataset_type": dataset_type,
        "requires_undo": False,
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "path_data": [os.path.join(__data_testing_dir__)],
        "target_suffix": target_lst,
        "extensions": [".nii.gz"],
        "roi_params": {"suffix": None, "slice_filter_roi": None},
        "model_params": {"name": "Unet", "is_2d": False, "length_3D": [32, 32, 32], "stride_3D": [32, 32, 32]},
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "patch_filter_params": patch_filter_params,
        "slice_axis": "axial",
        "multichannel": False
    }
    # Get Training dataset
    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)
    ds = imed_loader.load_dataset(bids_df, **loader_params)

    logger.info(f"\tNumber of loaded subvolumes: {len(ds)}")

    loader = DataLoader(ds, batch_size=BATCH_SIZE,
                        shuffle=True, pin_memory=True,
                        collate_fn=imed_loader_utils.imed_collate,
                        num_workers=0)
    logger.info("\tNumber of Neg/Pos subvolumes in GT.")
    cmpt_neg, cmpt_pos = _cmpt_slice(loader)
    if patch_filter_params["filter_empty_mask"]:
        if dataset_type == "testing":
            # Filters on patches are not applied at testing time
            assert cmpt_neg + cmpt_pos == len(ds)
        else:
            # Filters on patches are applied at training time
            assert cmpt_neg == 0
            assert cmpt_pos != 0
    else:
        # We verify if there are still some negative patches (they are removed with our filter)
        assert cmpt_neg != 0 and cmpt_pos != 0


def teardown_function():
    remove_tmp_dir()
