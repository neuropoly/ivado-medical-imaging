from __future__ import annotations

import copy
import typing
from typing import List, Tuple

from ivadomed.keywords import ROIParamsKW
from ivadomed.transforms import Compose, prepare_transforms

if typing.TYPE_CHECKING:
    from ivadomed.loader.patch_filter import PatchFilter
    from ivadomed.loader.slice_filter import SliceFilter


class GeneralizedLoaderConfiguration:
    """
    A generalized class to store ALL key data requirement for the loading purpose.
    It is a ONE-STOP configuration data class to enable subsequent downstream analyses.
    Used to significantly simply the constructor parameters
    """

    def __init__(
            self,
            # Mandatory Parameters
            model_params: dict,
            # Optional Parameters
            object_detection_params: dict = None,
            transform: list = None,
            nibabel_cache: bool = True,
            disk_cache: bool = True,
            metadata_choice: bool = False,
            multichannel: bool = False,
            soft_gt: bool = False,
            is_input_dropout: bool = False,
            slice_axis: int = 2,
            slice_filter_fn: callable = None,
            patch_filter_fn: callable = None,
            roi_params: dict = None,
            task: str = "segmentation",
            length: list or Tuple[int, int, int] = None,
            stride: list or Tuple[int, int, int] = None,
            # Bids specific
            bids_df: dict = None,
            subject_file_lst=None,
            target_suffix=None,
            contrast_params=None
    ):
        # Mandatory
        self.model_params: dict = model_params

        # Optional
        self.slice_axis: int = slice_axis
        self.nibabel_cache: bool = nibabel_cache

        # Default empty transform
        if transform is None:
            self.transform, self.UndoTransform = prepare_transforms({}, True)

        else:
            self.transform: list = transform

        self.metadata_choice: bool = metadata_choice
        self.slice_filter_fn: SliceFilter = slice_filter_fn
        self.patch_filter_fn: PatchFilter = patch_filter_fn
        self.roi_params: dict = (
            roi_params
            if roi_params is not None
            else {ROIParamsKW.SUFFIX: None, ROIParamsKW.SLICE_FILTER_ROI: None}
        )
        self.multichannel: bool = multichannel
        self.object_detection_params: dict = object_detection_params
        self.task: str = task
        # Whether to use soft ground truth or not
        self.soft_gt: bool = soft_gt
        self.is_input_dropout: bool = is_input_dropout
        self.disk_cache: bool = disk_cache
        self.length: typing.Tuple[int, int, int] = length
        self.stride: typing.Tuple[int, int, int] = stride

        # Optional Bids specific
        self.bids_df: dict = bids_df
        self.subject_file_lst: List[str] = subject_file_lst
        self.target_suffix: List[str] = target_suffix
        self.contrast_params: dict = contrast_params
