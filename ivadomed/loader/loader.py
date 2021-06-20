import copy
import random
import numpy as np
import torch
from loguru import logger

from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, adaptative as imed_adaptative
from ivadomed.loader.bids_dataset import BidsDataset
from ivadomed.loader.mri3d_subvolume_segmentation_dataset import MRI3DSubVolumeSegmentationDataset


def load_dataset(bids_df, data_list, transforms_params, model_params, target_suffix, roi_params,
                 contrast_params, slice_filter_params, slice_axis, multichannel,
                 dataset_type="training", requires_undo=False, metadata_type=None,
                 object_detection_params=None, soft_gt=False, device=None,
                 cuda_available=None, is_input_dropout=False, **kwargs):
    """Get loader appropriate loader according to model type. Available loaders are Bids3DDataset for 3D data,
    BidsDataset for 2D data and HDF5Dataset for HeMIS.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        data_list (list): Subject names list.
        transforms_params (dict): Dictionary containing transformations for "training", "validation", "testing" (keys),
            eg output of imed_transforms.get_subdatasets_transforms.
        model_params (dict): Dictionary containing model parameters.
        target_suffix (list of str): List of suffixes for target masks.
        roi_params (dict): Contains ROI related parameters.
        contrast_params (dict): Contains image contrasts related parameters.
        slice_filter_params (dict): Contains slice_filter parameters, see :doc:`configuration_file` for more details.
        slice_axis (string): Choice between "axial", "sagittal", "coronal" ; controls the axis used to extract the 2D
            data from 3D NifTI files. 2D PNG/TIF/JPG files use default "axial.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        metadata_type (str): Choice between None, "mri_params", "contrasts".
        dataset_type (str): Choice between "training", "validation" or "testing".
        requires_undo (bool): If True, the transformations without undo_transform will be discarded.
        object_detection_params (dict): Object dection parameters.
        soft_gt (bool): If True, ground truths are not binarized before being fed to the network. Otherwise, ground
        truths are thresholded (0.5) after the data augmentation operations.
        is_input_dropout (bool): Return input with missing modalities.

    Returns:
        BidsDataset

    Note: For more details on the parameters transform_params, target_suffix, roi_params, contrast_params,
    slice_filter_params and object_detection_params see :doc:`configuration_file`.
    """
    # Compose transforms
    tranform_lst, _ = imed_transforms.prepare_transforms(copy.deepcopy(transforms_params), requires_undo)

    # If ROICrop is not part of the transforms, then enforce no slice filtering based on ROI data.
    if 'ROICrop' not in transforms_params:
        roi_params["slice_filter_roi"] = None

    if model_params["name"] == "Modified3DUNet" or ('is_2d' in model_params and not model_params['is_2d']):
        dataset = Bids3DDataset(bids_df=bids_df,
                                subject_file_lst=data_list,
                                target_suffix=target_suffix,
                                roi_params=roi_params,
                                contrast_params=contrast_params,
                                metadata_choice=metadata_type,
                                slice_axis=imed_utils.AXIS_DCT[slice_axis],
                                transform=tranform_lst,
                                multichannel=multichannel,
                                model_params=model_params,
                                object_detection_params=object_detection_params,
                                soft_gt=soft_gt,
                                is_input_dropout=is_input_dropout)

    elif model_params["name"] == "HeMISUnet":
        dataset = imed_adaptative.HDF5Dataset(bids_df=bids_df,
                                              subject_file_lst=data_list,
                                              model_params=model_params,
                                              contrast_params=contrast_params,
                                              target_suffix=target_suffix,
                                              slice_axis=imed_utils.AXIS_DCT[slice_axis],
                                              transform=tranform_lst,
                                              metadata_choice=metadata_type,
                                              slice_filter_fn=imed_loader_utils.SliceFilter(**slice_filter_params,
                                                                                            device=device,
                                                                                            cuda_available=cuda_available),
                                              roi_params=roi_params,
                                              object_detection_params=object_detection_params,
                                              soft_gt=soft_gt)
    else:
        # Task selection
        task = imed_utils.get_task(model_params["name"])

        dataset = BidsDataset(bids_df=bids_df,
                              subject_file_lst=data_list,
                              target_suffix=target_suffix,
                              roi_params=roi_params,
                              contrast_params=contrast_params,
                              model_params=model_params,
                              metadata_choice=metadata_type,
                              slice_axis=imed_utils.AXIS_DCT[slice_axis],
                              transform=tranform_lst,
                              multichannel=multichannel,
                              slice_filter_fn=imed_loader_utils.SliceFilter(**slice_filter_params, device=device,
                                                                            cuda_available=cuda_available),
                              soft_gt=soft_gt,
                              object_detection_params=object_detection_params,
                              task=task,
                              is_input_dropout=is_input_dropout)
        dataset.load_filenames()

    if model_params["name"] == "Modified3DUNet":
        logger.info("Loaded {} volumes of shape {} for the {} set.".format(len(dataset), dataset.length, dataset_type))
    elif model_params["name"] != "HeMISUnet" and dataset.length:
        logger.info("Loaded {} {} patches of shape {} for the {} set.".format(len(dataset), slice_axis, dataset.length,
                                                                              dataset_type))
    else:
        logger.info("Loaded {} {} slices for the {} set.".format(len(dataset), slice_axis, dataset_type))

    return dataset


def dropout_input(seg_pair):
    """Applies input-level dropout: zero to all channels minus one will be randomly set to zeros. This function verifies
    if some channels are already empty. Always at least one input channel will be kept.

    Args:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata.

    Return:
        seg_pair (dict): Batch containing torch tensors (input and gt) and metadata with channel(s) dropped.
    """
    n_channels = seg_pair['input'].size(0)
    # Verify if the input is multichannel
    if n_channels > 1:
        # Verify if some channels are already empty
        n_unique_values = [len(torch.unique(input_data)) > 1 for input_data in seg_pair['input']]
        idx_empty = np.where(np.invert(n_unique_values))[0]

        # Select how many channels will be dropped between 0 and n_channels - 1 (keep at least one input)
        n_dropped = random.randint(0, n_channels - 1)

        if n_dropped > len(idx_empty):
            # Remove empty channel to the number of channels to drop
            n_dropped = n_dropped - len(idx_empty)
            # Select which channels will be dropped
            idx_dropped = []
            while len(idx_dropped) != n_dropped:
                idx = random.randint(0, n_channels - 1)
                # Don't include the empty channel in the dropped channels
                if idx not in idx_empty:
                    idx_dropped.append(idx)
        else:
            idx_dropped = idx_empty

        seg_pair['input'][idx_dropped] = torch.zeros_like(seg_pair['input'][idx_dropped])

    else:
        logger.warning("\n Impossible to apply input-level dropout since input is not multi-channel.")

    return seg_pair


class Bids3DDataset(MRI3DSubVolumeSegmentationDataset):
    """BIDS specific dataset loader for 3D dataset.

    Args:
        bids_df (BidsDataframe): Object containing dataframe with all BIDS image files and their metadata.
        subject_file_lst (list): Subject filenames list.
        target_suffix (list): List of suffixes for target masks.
        model_params (dict): Dictionary containing model parameters.
        contrast_params (dict): Contains image contrasts related parameters.
        slice_axis (int): Indicates the axis used to extract slices: "axial": 2, "sagittal": 0, "coronal": 1.
        cache (bool): If the data should be cached in memory or not.
        transform (list): Transformation list (length 2) composed of preprocessing transforms (Compose) and transforms
            to apply during training (Compose).
        metadata_choice: Choice between "mri_params", "contrasts", None or False, related to FiLM.
        roi_params (dict): Dictionary containing parameters related to ROI image processing.
        multichannel (bool): If True, the input contrasts are combined as input channels for the model. Otherwise, each
            contrast is processed individually (ie different sample / tensor).
        object_detection_params (dict): Object dection parameters.
        is_input_dropout (bool): Return input with missing modalities.
    """

    def __init__(self, bids_df, subject_file_lst, target_suffix, model_params, contrast_params, slice_axis=2,
                 cache=True, transform=None, metadata_choice=False, roi_params=None,
                 multichannel=False, object_detection_params=None, task="segmentation", soft_gt=False,
                 is_input_dropout=False):
        dataset = BidsDataset(bids_df=bids_df,
                              subject_file_lst=subject_file_lst,
                              target_suffix=target_suffix,
                              roi_params=roi_params,
                              contrast_params=contrast_params,
                              model_params=model_params,
                              metadata_choice=metadata_choice,
                              slice_axis=slice_axis,
                              transform=transform,
                              multichannel=multichannel,
                              object_detection_params=object_detection_params,
                              is_input_dropout=is_input_dropout)

        super().__init__(dataset.filename_pairs, length=model_params["length_3D"], stride=model_params["stride_3D"],
                         transform=transform, slice_axis=slice_axis, task=task, soft_gt=soft_gt,
                         is_input_dropout=is_input_dropout)


