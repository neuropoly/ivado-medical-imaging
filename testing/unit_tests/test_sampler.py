import numpy as np
import pytest
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from loguru import logger

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
from ivadomed.loader.balanced_sampler import BalancedSampler
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir


cudnn.benchmark = True
GPU_ID = 0
BATCH_SIZE = 1


def setup_function():
    create_tmp_dir()


def _cmpt_label(ds_loader):
    cmpt_label, cmpt_sample = {0: 0, 1: 0}, 0
    for i, batch in enumerate(ds_loader):
        for gt in batch['gt']:
            if np.any(gt.numpy()):
                cmpt_label[1] += 1
            else:
                cmpt_label[0] += 1
        cmpt_sample += len(batch['gt'])

    neg_sample_ratio = cmpt_label[0] * 100. / cmpt_sample
    pos_sample_ratio = cmpt_label[1] * 100. / cmpt_sample
    logger.info({'neg_sample_ratio': neg_sample_ratio,
           'pos_sample_ratio': pos_sample_ratio})
    return neg_sample_ratio, pos_sample_ratio


@pytest.mark.parametrize('transforms_dict', [{
    "Resample":
        {
            "wspace": 0.75,
            "hspace": 0.75
        },
    "ROICrop":
        {
            "size": [128, 128]
        },
    "NumpyToTensor": {}
}])
@pytest.mark.parametrize('train_lst', [['sub-unf01_T2w.nii.gz']])
@pytest.mark.parametrize('target_lst', [["_lesion-manual"]])
@pytest.mark.parametrize('roi_params', [{"suffix": "_seg-manual", "slice_filter_roi": 10}])
def test_sampler(download_data_testing_test_files, transforms_dict, train_lst, target_lst, roi_params):
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
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
        "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "slice_axis": "axial",
        "multichannel": False
    }
    # Get Training dataset
    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)
    ds_train = imed_loader.load_dataset(bids_df, **loader_params)

    logger.info("\nLoading without sampling")
    train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=0)
    neg_percent, pos_percent = _cmpt_label(train_loader)
    assert abs(neg_percent - pos_percent) > 20

    logger.info("\nLoading with sampling")
    train_loader_balanced = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                       sampler=BalancedSampler(ds_train),
                                       shuffle=False, pin_memory=True,
                                       collate_fn=imed_loader_utils.imed_collate,
                                       num_workers=0)

    neg_percent_bal, pos_percent_bal = _cmpt_label(train_loader_balanced)
    # Check if the loader is more balanced. The actual distribution comes from a probabilistic model
    # This is however not very efficient to get close to 50 %
    # in the case where we have 16 slices, with 87.5 % of one class (positive sample).
    assert abs(neg_percent_bal - pos_percent_bal) <= abs(neg_percent - pos_percent)


def teardown_function():
    remove_tmp_dir()
