import pytest
import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from loguru import logger

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed.loader.bids3d_dataset import Bids3DDataset
from ivadomed.loader.bids_dataset import BidsDataset
from ivadomed.loader.segmentation_pair import SegmentationPair
from ivadomed import metrics as imed_metrics
from ivadomed import postprocessing as imed_postpro
from ivadomed import transforms as imed_transforms
from ivadomed.loader import loader as imed_loader, utils as imed_loader_utils
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir


GPU_ID = 0


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [__data_testing_dir__],
    "target_suffix": ["_seg-manual"],
    "extensions": [".nii.gz"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ['T1w'],  "balance": {}}
    }])
def test_image_orientation(download_data_testing_test_files, loader_parameters):
    device = torch.device("cuda:" + str(GPU_ID) if torch.cuda.is_available() else "cpu")
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.set_device(device)
        logger.info(f"Using GPU ID {device}")

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)

    contrast_params = loader_parameters["contrast_params"]
    target_suffix = loader_parameters["target_suffix"]
    roi_params = loader_parameters["roi_params"]

    train_lst = ['sub-unf01_T1w.nii.gz']

    training_transform_dict = {
        "Resample":
            {
                "wspace": 1.5,
                "hspace": 1,
                "dspace": 3
            },
        "CenterCrop":
            {
                "size": [176, 128, 160]
            },
        "NormalizeInstance": {"applied_to": ['im']}
    }

    tranform_lst, training_undo_transform = imed_transforms.prepare_transforms(training_transform_dict)

    model_params = {
            "name": "Modified3DUNet",
            "dropout_rate": 0.3,
            "bn_momentum": 0.9,
            "depth": 2,
            "in_channel": 1,
            "out_channel": 1,
            "length_3D": [176, 128, 160],
            "stride_3D": [176, 128, 160],
            "attention": False,
            "n_filters": 8
        }

    for dim in ['2d', '3d']:
        for slice_axis in [0, 1, 2]:
            if dim == '2d':
                ds = BidsDataset(bids_df=bids_df,
                                 subject_file_lst=train_lst,
                                 target_suffix=target_suffix,
                                 contrast_params=contrast_params,
                                 model_params=model_params,
                                 metadata_choice=False,
                                 slice_axis=slice_axis,
                                 transform=tranform_lst,
                                 multichannel=False)
                ds.load_filenames()
            else:
                ds = Bids3DDataset(bids_df=bids_df,
                                   subject_file_lst=train_lst,
                                   target_suffix=target_suffix,
                                   model_params=model_params,
                                   contrast_params=contrast_params,
                                   metadata_choice=False,
                                   slice_axis=slice_axis,
                                   transform=tranform_lst,
                                   multichannel=False)

            loader = DataLoader(ds, batch_size=1,
                                shuffle=True, pin_memory=True,
                                collate_fn=imed_loader_utils.imed_collate,
                                num_workers=1)

            input_filename, gt_filename, roi_filename, metadata = ds.filename_pairs[0]
            segpair = SegmentationPair(input_filename, gt_filename, metadata=metadata,
                                                                         slice_axis=slice_axis)
            nib_original = nib.load(gt_filename[0])
            # Get image with original, ras and hwd orientations
            input_init = nib_original.get_fdata()
            input_ras = nib.as_closest_canonical(nib_original).get_fdata()
            img, gt = segpair.get_pair_data()
            input_hwd = gt[0]

            pred_tmp_lst, z_tmp_lst = [], []
            for i, batch in enumerate(loader):
                # batch["input_metadata"] = batch["input_metadata"][0]  # Take only metadata from one input
                # batch["gt_metadata"] = batch["gt_metadata"][0]  # Take only metadata from one label

                for smp_idx in range(len(batch['gt'])):
                    # undo transformations
                    if dim == '2d':
                        preds_idx_undo, metadata_idx = training_undo_transform(batch["gt"][smp_idx],
                                                                               batch["gt_metadata"][smp_idx],
                                                                               data_type='gt')

                        # add new sample to pred_tmp_lst
                        pred_tmp_lst.append(preds_idx_undo[0])
                        z_tmp_lst.append(int(batch['input_metadata'][smp_idx][0]['slice_index']))

                    else:
                        preds_idx_undo, metadata_idx = training_undo_transform(batch["gt"][smp_idx],
                                                                               batch["gt_metadata"][smp_idx],
                                                                               data_type='gt')

                    fname_ref = metadata_idx[0]['gt_filenames'][0]

                    if (pred_tmp_lst and i == len(loader) - 1) or dim == '3d':
                        # save the completely processed file as a nii
                        nib_ref = nib.load(fname_ref)
                        nib_ref_can = nib.as_closest_canonical(nib_ref)

                        if dim == '2d':
                            tmp_lst = []
                            for z in range(nib_ref_can.header.get_data_shape()[slice_axis]):
                                tmp_lst.append(pred_tmp_lst[z_tmp_lst.index(z)])
                            arr = np.stack(tmp_lst, axis=-1)
                        else:
                            arr = np.array(preds_idx_undo[0])

                        # verify image after transform, undo transform and 3D reconstruction
                        input_hwd_2 = imed_postpro.threshold_predictions(arr)
                        # Some difference are generated due to transform and undo transform
                        # (e.i. Resample interpolation)
                        assert imed_metrics.dice_score(input_hwd_2, input_hwd) >= 0.8
                        input_ras_2 = imed_loader_utils.orient_img_ras(input_hwd_2, slice_axis)
                        assert imed_metrics.dice_score(input_ras_2, input_ras) >= 0.8
                        input_init_2 = imed_loader_utils.reorient_image(input_hwd_2, slice_axis, nib_ref, nib_ref_can)
                        assert imed_metrics.dice_score(input_init_2, input_init) >= 0.8

                        # re-init pred_stack_lst
                        pred_tmp_lst, z_tmp_lst = [], []


def teardown_function():
    remove_tmp_dir()
