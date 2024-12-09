#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.postprocessing


import nibabel as nib
import numpy as np
import pytest
import scipy
from ivadomed import postprocessing as imed_postpro
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path


def setup_function():
    create_tmp_dir()


def nii_dummy_seg(size_arr=(15, 15, 9), pixdim=(1, 1, 1), dtype=np.float64, orientation='LPI',
                  shape='rectangle', radius_RL=3.0, radius_AP=2.0, zeroslice=None, softseg=False):
    """Create a dummy nibabel object.

    Create either an ellipse or rectangle of ones running from top to bottom in the 3rd
    dimension.

    Args:
        size_arr (tuple): (nx, ny, nz)
        pixdim (tuple): (px, py, pz)
        dtype: Numpy dtype.
        orientation: Orientation of the image. Default: LPI
        shape: {'rectangle', 'ellipse'}
        radius_RL (float): 1st radius. With a, b = 50.0, 30.0 (in mm), theoretical CSA of ellipse
            is 4712.4
        radius_AP: float: 2nd radius
        zeroslice (list int): zero all slices listed in this param
        softseg (bool): Generate soft segmentation by applying blurring filter.

    Retunrs:
        nibabel: Image object
    """
    if zeroslice is None:
        zeroslice = []
    # Create a 3d array, with dimensions corresponding to x: RL, y: AP, z: IS
    nx, ny, nz = [int(size_arr[i] * pixdim[i]) for i in range(3)]
    data = np.zeros((nx, ny, nz), dtype)
    xx, yy = np.mgrid[:nx, :ny]
    # loop across slices and add object
    for iz in range(nz):
        if shape == 'rectangle':  # theoretical CSA: (a*2+1)(b*2+1)
            data[:, :, iz] = ((abs(xx - nx / 2) <= radius_RL) & (abs(yy - ny / 2) <= radius_AP)) * 1
        if shape == 'ellipse':
            data[:, :, iz] = (((xx - nx / 2) / radius_RL) ** 2 + ((yy - ny / 2) / radius_AP) ** 2 <= 1) * 1
    # Zero specified slices
    if zeroslice is not []:
        data[:, :, zeroslice] = 0
    # Apply Gaussian filter (to get soft seg)
    if softseg:
        kernel = np.ones((3, 3, 3)) / 27
        data = scipy.ndimage.convolve(data, kernel)
    # Create nibabel object
    affine = np.eye(4)
    nii = nib.nifti1.Nifti1Image(data, affine)
    # Change orientation
    # TODO
    return nii


def check_bin_vs_soft(arr_in, arr_out):
    """Make sure that if input was bin, output is also bin. Or, if input was soft, output is soft.

    Args:
        arr_in: TODO
        arr_out: TODO
    """
    if np.array_equal(arr_in, arr_in.astype(bool)):
        if np.array_equal(arr_out, arr_out.astype(bool)):
            # Both arr_in and arr_out are bin
            return True
        else:
            return False
    else:
        if np.array_equal(arr_out, arr_out.astype(bool)):
            return False
        else:
            # Both arr_in and arr_out are soft
            return True


@pytest.mark.parametrize('nii_seg', [nii_dummy_seg(softseg=True)])
def test_threshold(nii_seg):
    # input array
    arr_seg_proc = imed_postpro.threshold_predictions(np.copy(np.asanyarray(nii_seg.dataobj)))
    assert isinstance(arr_seg_proc, np.ndarray)
    # Before thresholding: [0.33333333, 0.66666667, 1.        ] --> after thresholding: [0, 1, 1]
    assert np.array_equal(arr_seg_proc[4:7, 8, 4], np.array([0, 1, 1]))
    # input nibabel
    nii_seg_proc = imed_postpro.threshold_predictions(nii_seg)
    assert isinstance(nii_seg_proc, nib.nifti1.Nifti1Image)
    assert np.array_equal(nii_seg_proc.get_fdata()[4:7, 8, 4], np.array([0, 1, 1]))


@pytest.mark.parametrize('nii_seg', [nii_dummy_seg(), nii_dummy_seg(softseg=True)])
def test_keep_largest_object(nii_seg):
    # Set a voxel to 1 at the corner to make sure it is set to 0 by the function
    coord = (1, 1, 1)
    nii_seg.dataobj[coord] = 1
    # Test function with array input
    arr_seg_proc = imed_postpro.keep_largest_object(np.copy(np.asanyarray(nii_seg.dataobj)))
    assert isinstance(arr_seg_proc, np.ndarray)
    assert check_bin_vs_soft(nii_seg.dataobj, arr_seg_proc)
    assert arr_seg_proc[coord] == 0
    # Make sure it works with nibabel input
    nii_seg_proc = imed_postpro.keep_largest_object(nii_seg)
    assert isinstance(nii_seg_proc, nib.nifti1.Nifti1Image)
    assert check_bin_vs_soft(nii_seg.dataobj, nii_seg_proc.dataobj)
    assert nii_seg_proc.dataobj[coord] == 0


@pytest.mark.parametrize('nii_seg', [nii_dummy_seg(), nii_dummy_seg(softseg=True)])
def test_keep_largest_object_per_slice(nii_seg):
    # Set a voxel to 1 at the corner to make sure it is set to 0 by the function
    coord = (1, 1, 1)
    nii_seg.dataobj[coord] = 1
    # Test function with array input
    arr_seg_proc = imed_postpro.keep_largest_object_per_slice(np.copy(
                                                                np.asanyarray(nii_seg.dataobj)),
                                                              axis=2)
    assert isinstance(arr_seg_proc, np.ndarray)
    assert check_bin_vs_soft(nii_seg.dataobj, arr_seg_proc)
    assert arr_seg_proc[coord] == 0
    # Make sure it works with nibabel input
    nii_seg_proc = imed_postpro.keep_largest_object_per_slice(nii_seg)
    assert isinstance(nii_seg_proc, nib.nifti1.Nifti1Image)
    assert check_bin_vs_soft(nii_seg.dataobj, nii_seg_proc.dataobj)
    assert nii_seg_proc.dataobj[coord] == 0


@pytest.mark.parametrize('nii_seg', [nii_dummy_seg()])
def test_fill_holes(nii_seg):
    # Set a voxel to 0 in the middle of the segmentation to make sure it is set to 1 by the function
    coord = (7, 7, 4)
    nii_seg.dataobj[coord] = 0
    # Test function with array input
    arr_seg_proc = imed_postpro.fill_holes(np.copy(np.asanyarray(nii_seg.dataobj)))
    assert isinstance(arr_seg_proc, np.ndarray)
    assert arr_seg_proc[coord] == 1
    # Make sure it works with nibabel input
    nii_seg_proc = imed_postpro.fill_holes(nii_seg)
    assert isinstance(nii_seg_proc, nib.nifti1.Nifti1Image)
    assert nii_seg_proc.dataobj[coord] == 1


@pytest.mark.parametrize('nii_seg', [nii_dummy_seg()])
def test_mask_predictions(nii_seg):
    # create nii object with a voxel of 0 somewhere in the middle
    nii_seg_mask = nib.nifti1.Nifti1Image(np.copy(np.asanyarray(nii_seg.dataobj)), nii_seg.affine)
    coord = (7, 7, 4)
    nii_seg_mask.dataobj[coord] = 0
    # Test function with array input
    arr_seg_proc = imed_postpro.mask_predictions(
        np.copy(np.asanyarray(nii_seg.dataobj)), np.asanyarray(nii_seg_mask.dataobj))
    assert isinstance(arr_seg_proc, np.ndarray)
    assert arr_seg_proc[coord] == 0
    # Make sure it works with nibabel input
    nii_seg_proc = imed_postpro.mask_predictions(nii_seg, nii_seg_mask.dataobj)
    assert isinstance(nii_seg_proc, nib.nifti1.Nifti1Image)
    assert nii_seg_proc.dataobj[coord] == 0


def test_label_file_from_coordinates(download_data_testing_test_files):
    # create fake coordinate
    coord = [[0, 0, 0]]
    # load test image
    nifti = nib.load(
        Path(__data_testing_dir__, 'sub-unf01/anat/sub-unf01_T1w.nii.gz'))
    # create fake label
    label = imed_postpro.label_file_from_coordinates(nifti, coord)
    # check if it worked
    assert isinstance(label, nib.nifti1.Nifti1Image)


def teardown_function():
    remove_tmp_dir()
