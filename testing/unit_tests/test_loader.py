import os
from pathlib import Path
import shutil

import pytest
import csv_diff
import torch
import numpy as np
from loguru import logger

from ivadomed.loader.bids_dataframe import BidsDataframe
from testing.unit_tests.t_utils import create_tmp_dir, __data_testing_dir__, __tmp_dir__, download_data_testing_test_files, path_repo_root
from testing.common_testing_util import remove_tmp_dir
from ivadomed import utils as imed_utils
from ivadomed.loader import loader as imed_loader
import ivadomed.loader.utils as imed_loader_utils
from ivadomed.loader import mri2d_segmentation_dataset as imed_loader_mri2dseg
from ivadomed.keywords import LoaderParamsKW, MetadataKW, ModelParamsKW, TransformationKW



def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
    "target_suffix": [["_seg-myelin-manual", "_seg-axon-manual"]],
    "extensions": [".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": []}
}])
def test_bids_df_microscopy_png(download_data_testing_test_files, loader_parameters):
    """
    Test for microscopy png file format
    Test for _sessions.tsv and _scans.tsv files
    Test for target_suffix as a nested list
    Test for when no contrast_params are provided
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_ref.csv")
    csv_test = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [__data_testing_dir__],
    "target_suffix": ["_lesion-manual"],
    "extensions": [],
    "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ["T1w", "T2w"]}
    }])
def test_bids_df_anat(download_data_testing_test_files, loader_parameters):
    """
    Test for MRI anat nii.gz file format
    Test for when no file extensions are provided
    Test for multiple target_suffix
    Test behavior when "roi_suffix" is not None
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_ref.csv")
    csv_test = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "ct_scan"))],
    "bids_config": f"{path_repo_root}/ivadomed/config/config_bids.json",
    "target_suffix": ["_seg-manual"],
    "extensions": [".nii.gz"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ["ct"]},
    "bids_validate": False
    }])
def test_bids_df_no_validate(download_data_testing_test_files, loader_parameters):
    """
    Test for ct-scan nii.gz file format
    Test for when validate_BIDS is set to False for the loader
    """

    # Rename files so the loader won't pick them up if validate_BIDS is true
    Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "sub-spleen2").rename(
         Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "ssub-spleen2"))

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_ref.csv")
    csv_test = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(
        csv_diff.load_csv(open(csv_ref)),
        csv_diff.load_csv(open(csv_test))
    )

    Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "ssub-spleen2").rename(
         Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "sub-spleen2"))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [__data_testing_dir__, str(Path(__data_testing_dir__, "microscopy_png"))],
    "target_suffix": ["_seg-manual", "seg-axon-manual"],
    "extensions": [".nii.gz", ".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ["T1w", "T2w", "SEM"]}
    }])
def test_bids_df_multi(download_data_testing_test_files, loader_parameters):
    """
    Test for multiple folders in path_data
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_ref_multi.csv")
    csv_test = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_test_multi.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [],
                    'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "ct_scan"))],
    "bids_config": f"{path_repo_root}/ivadomed/config/config_bids.json",
    "target_suffix": ["_seg-manual"],
    "extensions": [".nii.gz"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": ["ct"]}
    }])
def test_bids_df_ctscan(download_data_testing_test_files, loader_parameters):
    """
    Test for ct-scan nii.gz file format
    Test for when dataset_description.json is not present in derivatives folder
    """

    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    df_test = bids_df.df.drop(columns=['path'])
    df_test = df_test.sort_values(by=['filename']).reset_index(drop=True)
    csv_ref = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_ref.csv")
    csv_test = Path(loader_parameters[LoaderParamsKW.PATH_DATA][0], "df_test.csv")
    df_test.to_csv(csv_test, index=False)
    diff = csv_diff.compare(csv_diff.load_csv(open(csv_ref)), csv_diff.load_csv(open(csv_test)))
    assert diff == {'added': [], 'removed': [], 'changed': [], 'columns_added': [], 'columns_removed': []}


@pytest.mark.parametrize('seg_pair', [
    {"input": torch.rand((2, 5, 5))},
    {"input": torch.rand((1, 5, 5))},
    {"input": torch.rand((5, 5, 5, 5))},
    {"input": (torch.rand((5, 5, 5, 3)) * torch.tensor([1, 0, 1], dtype=torch.float)).transpose(0, -1)},
    {"input": (torch.rand((7, 7, 4)) * torch.tensor([1, 0, 0, 0], dtype=torch.float)).transpose(0, -1)}
])
def test_dropout_input(seg_pair):
    n_channels = seg_pair['input'].size(0)
    seg_pair = imed_loader_utils.dropout_input(seg_pair)
    empty_channels = [len(torch.unique(input_data)) == 1 for input_data in seg_pair['input']]

    # If multichannel
    if n_channels > 1:
        # Verify that there is still at least one channel remaining
        assert sum(empty_channels) <= n_channels
    else:
        assert sum(empty_channels) == 0


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
    "target_suffix": ["_seg-myelin-manual"],
    "extensions": [".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": [], "balance": {}},
    "slice_axis": "axial",
    "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
    "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
    "multichannel": False
    }])
@pytest.mark.parametrize('model_parameters', [{
    "name": "Unet",
    "dropout_rate": 0.3,
    "bn_momentum": 0.1,
    "final_activation": "sigmoid",
    "depth": 3
    }])
@pytest.mark.parametrize('transform_parameters', [{
    "NumpyToTensor": {},
    }])
def test_load_dataset_2d_png(download_data_testing_test_files,
                             loader_parameters, model_parameters, transform_parameters):
    """
    Test to make sure load_dataset runs with 2D PNG files, writes corresponding NIfTI files,
    and binarizes ground-truth values to 0 and 1.
    """
    loader_parameters.update({LoaderParamsKW.MODEL_PARAMS: model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})
    fname_png = bids_df.df[bids_df.df['filename'] == data_lst[0]]['path'].values[0]
    fname_nii = imed_loader_utils.update_filename_to_nifti(fname_png)
    assert Path(fname_nii).exists() == 1
    assert ds[0]['input'].shape == (1, 756, 764)
    assert ds[0]['gt'].shape == (1, 756, 764)
    assert np.unique(ds[0]['gt']).tolist() == [0, 1]


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
    "target_suffix": ["_seg-myelin-manual"],
    "extensions": [".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": [], "balance": {}},
    "slice_axis": "axial",
    "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
    "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
    "multichannel": False
    }])
@pytest.mark.parametrize('model_parameters', [{
    "name": "Unet",
    "dropout_rate": 0.3,
    "bn_momentum": 0.1,
    "final_activation": "sigmoid",
    "depth": 3,
    "length_2D": [256, 128],
    "stride_2D": [244, 116]
    }])
@pytest.mark.parametrize('transform_parameters', [{
    "Resample": {
        "wspace": 0.0002,
        "hspace": 0.0001
    },
    "NumpyToTensor": {},
    }])
def test_2d_patches_and_resampling(download_data_testing_test_files,
                                   loader_parameters, model_parameters, transform_parameters):
    """
    Test that 2d patching is done properly.
    Test that microscopy pixelsize and resampling are applied on the right dimensions.
    """
    loader_parameters.update({LoaderParamsKW.MODEL_PARAMS: model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})
    assert ds.is_2d_patch == True
    assert ds[0]['input'].shape == (1, 256, 128)
    assert ds[0]['input_metadata'][0].metadata[MetadataKW.INDEX_SHAPE] == (1512, 382)
    assert len(ds) == 28


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
    "target_suffix": ["_seg-myelin-manual", "_seg-axon-manual"],
    "extensions": [".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": [], "balance": {}},
    "slice_axis": "axial",
    "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
    "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
    "multichannel": False
    }])
@pytest.mark.parametrize('model_parameters', [{
    "name": "Unet",
    "dropout_rate": 0.3,
    "bn_momentum": 0.1,
    "depth": 2
    }])
@pytest.mark.parametrize('transform_parameters', [{
    "NumpyToTensor": {},
    }])
def test_get_target_filename_list(loader_parameters, model_parameters, transform_parameters):
    """
    Test that all target_suffix are considered for target filename when list
    """
    loader_parameters.update({LoaderParamsKW.MODEL_PARAMS: model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    test_ds = imed_loader.load_dataset(bids_df,
                                       **{**loader_parameters, **{'data_list': data_lst,
                                                                  'transforms_params': transform_parameters,
                                                                  'dataset_type': 'training'}})
    target_filename = test_ds.filename_pairs[0][1]
    
    assert len(target_filename) == len(loader_parameters[LoaderParamsKW.TARGET_SUFFIX])


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
    "target_suffix": [["_seg-myelin-manual", "_seg-axon-manual"], ["_seg-myelin-manual", "_seg-axon-manual"]],
    "extensions": [".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": [], "balance": {}},
    "slice_axis": "axial",
    "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
    "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
    "multichannel": False
    }])
@pytest.mark.parametrize('model_parameters', [{
    "name": "Unet",
    "dropout_rate": 0.3,
    "bn_momentum": 0.1,
    "depth": 2
    }])
@pytest.mark.parametrize('transform_parameters', [{
    "NumpyToTensor": {},
    }])
def test_get_target_filename_list_multiple_raters(loader_parameters, model_parameters, transform_parameters):
    """
    Test that all target_suffix are considered for target filename when list
    """
    loader_parameters.update({LoaderParamsKW.MODEL_PARAMS: model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)
    data_lst = ['sub-rat3_ses-01_sample-data9_SEM.png']
    test_ds = imed_loader.load_dataset(bids_df,
                                       **{**loader_parameters, **{'data_list': data_lst,
                                                                  'transforms_params': transform_parameters,
                                                                  'dataset_type': 'training'}})
    target_filename = test_ds.filename_pairs[0][1]

    assert len(target_filename) == len(loader_parameters[LoaderParamsKW.TARGET_SUFFIX])
    assert len(target_filename[0]) == len(loader_parameters[LoaderParamsKW.TARGET_SUFFIX][0])
    assert len(target_filename[1]) == len(loader_parameters[LoaderParamsKW.TARGET_SUFFIX][1])


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
    "target_suffix": ["_seg-myelin-manual"],
    "extensions": [".png"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": [], "balance": {}},
    "slice_axis": "axial",
    "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
    "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
    "multichannel": False
    }])
@pytest.mark.parametrize('model_parameters', [{
    "name": "Unet",
    "dropout_rate": 0.3,
    "bn_momentum": 0.1,
    "final_activation": "sigmoid",
    "depth": 3
    }])
def test_microscopy_pixelsize(download_data_testing_test_files, loader_parameters, model_parameters):
    """
    Test that PixelSize and PixelSizeUnits microscopy metadata
    are handled properly for PixelSizeUnits: "mm", "um" and "nm"
    """
    loader_parameters.update({LoaderParamsKW.MODEL_PARAMS: model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=True)

    # PixelSizeUnits: "mm"
    data_lst = ['sub-rat2_sample-data5_SEM.png']
    transform_parameters = {TransformationKW.RESAMPLE: {"wspace": 0.000093, "hspace": 0.000093}}
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})
    assert ds[0]['input'].shape == (1, 725, 725)

    # PixelSizeUnits: "um"
    data_lst = ['sub-rat3_ses-02_sample-data11_run-1_SEM.png']
    transform_parameters = {TransformationKW.RESAMPLE: {"wspace": 0.0001, "hspace": 0.0001}}
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})
    assert ds[0]['input'].shape == (1, 839, 769)

    # PixelSizeUnits: "nm"
    data_lst = ['sub-rat3_ses-02_sample-data10_SEM.png']
    transform_parameters = {TransformationKW.RESAMPLE: {"wspace": 0.0001, "hspace": 0.0001}}
    ds = imed_loader.load_dataset(bids_df,
                                  **{**loader_parameters, **{'data_list': data_lst,
                                                             'transforms_params': transform_parameters,
                                                             'dataset_type': 'training'}})
    assert ds[0]['input'].shape == (1, 758, 737)


@pytest.mark.parametrize('loader_parameters', [{
    "path_data": [str(Path(__data_testing_dir__, "data_test_png_tif"))],
    "target_suffix": ["_seg-myelin-manual"],
    "extensions": [".png", ".tif"],
    "roi_params": {"suffix": None, "slice_filter_roi": None},
    "contrast_params": {"contrast_lst": [], "balance": {}},
    "slice_axis": "axial",
    "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
    "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
    "multichannel": False
    }])
@pytest.mark.parametrize('model_parameters', [{
    "name": "Unet",
    "dropout_rate": 0.3,
    "bn_momentum": 0.1,
    "final_activation": "sigmoid",
    "depth": 3
    }])
def test_read_png_tif(download_data_testing_test_files, loader_parameters, model_parameters):
    """
    Test to make sure all combinaitions of PNG/TIF, 8/16 bits, Grayscale/RGB/RGBA files
    can be loaded without errors.
    """
    metadata = {}
    metadata[MetadataKW.PIXEL_SIZE] = [0.07, 0.07]
    metadata[MetadataKW.PIXEL_SIZE_UNITS] = "um"
    loader_parameters.update({LoaderParamsKW.MODEL_PARAMS: model_parameters})
    bids_df = BidsDataframe(loader_parameters, __tmp_dir__, derivatives=False)
    file_lst = bids_df.df['path'].tolist()
    filename_pairs = [(file_lst, None, None, metadata if isinstance(metadata, list) else [metadata])]
    slice_axis = imed_utils.AXIS_DCT[loader_parameters[LoaderParamsKW.SLICE_AXIS]]
    ds = imed_loader_mri2dseg.MRI2DSegmentationDataset(filename_pairs,
                                                       slice_axis=slice_axis,
                                                       nibabel_cache=True,
                                                       transform=[None, None],
                                                       slice_filter_fn=None)
    ds.load_filenames()

def test_create_cache_folder():
    """
    Test to make sure the cache folder is created when it doesn't exist, remove it afterwards.
    NOTE: this means this test cannot be parallelized with other tests that utilize this folder!
    """
    path_cache = imed_loader_utils.create_temp_directory()
    print(path_cache)
    assert(os.path.exists(path_cache))
    shutil.rmtree(path_cache)

def teardown_function():
    remove_tmp_dir()
