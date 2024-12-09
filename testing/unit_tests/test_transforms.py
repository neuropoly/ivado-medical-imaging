#!/usr/bin/env python
# -*- coding: utf-8
# pytest unit tests for ivadomed.transforms


from math import isclose
import numpy as np
import pytest
import torch
from scipy.ndimage import center_of_mass, label
from ivadomed import maths as imed_maths

from ivadomed.loader.sample_meta_data import SampleMetadata
from ivadomed.metrics import dice_score
from ivadomed.transforms import Clahe, AdditiveGaussianNoise, RandomAffine, RandomReverse, \
    DilateGT, ElasticTransform, ROICrop, CenterCrop, NormalizeInstance, HistogramClipping, \
    NumpyToTensor, Resample
from ivadomed.keywords import MetadataKW

DEBUGGING = False
if DEBUGGING:
    from ivadomed.utils import plot_transformed_sample


def create_test_image(width, height, depth=0, num_contrasts=1, noise_max=10.0, num_objs=1, rad_max=30,
                      num_seg_classes=1, random_position=False):
    """Create test image.

    Create test image and its segmentation with a given number of objects, classes, and maximum radius.
    Compatible with both 2D (depth=0) and 3D images.

    Args:
        height (int): height image
        width (int): width image
        depth (int): depth image, if 0 then 2D images are returned
        num_contrasts (int): number of contrasts
        noise_max (float): noise from the uniform distribution [0,noise_max)
        num_objs (int): number of objects
        rad_max (int): maximum radius of objects
        num_seg_classes (int): number of classes
        random_position (bool): If false, the object is located at the center of the image. Otherwise, randomly located.

    Return:
        list, list: image and segmentation, list of num_contrasts elements of shape (height, width, depth).

    Adapted from: https://github.com/Project-MONAI/MONAI/blob/master/monai/data/synthetic.py#L17
    """
    assert num_contrasts >= 1

    depth_ = depth if depth >= 1 else 2 * rad_max + 1
    assert (height > 2 * rad_max) and (width > 2 * rad_max) and (depth_ > 2 * rad_max)

    image = np.zeros((height, width, depth_))

    for i in range(num_objs):
        if random_position:
            x = np.random.randint(rad_max, height - rad_max)
            y = np.random.randint(rad_max, width - rad_max)
            z = np.random.randint(rad_max, depth_ - rad_max)
        else:
            x, y, z = np.rint(height / 2), np.rint(width / 2), np.rint(depth_ / 2)
        rad = np.random.randint(5, rad_max)
        spy, spx, spz = np.ogrid[-x:height - x, -y:width - y, -z:depth_ - z]
        sphere = (spx * spx + spy * spy + spz * spz) <= rad * rad * rad

        if num_seg_classes > 1:
            image[sphere] = np.ceil(np.random.random() * num_seg_classes)
        else:
            image[sphere] = np.random.random() * 0.5 + 0.5

    seg = np.ceil(image).astype(np.int32)

    if depth == 0:
        _, _, z_slice = center_of_mass(seg.astype(int))
        z_slice = int(round(z_slice))
        seg = seg[:, :, z_slice]

    list_im, list_seg = [], []
    for _ in range(num_contrasts):
        norm = np.random.uniform(0, num_seg_classes * noise_max, size=image.shape)
        noisy_image = imed_maths.rescale_values_array(np.maximum(image, norm))

        if depth == 0:
            noisy_image = noisy_image[:, :, z_slice]

        list_im.append(noisy_image)
        list_seg.append(seg)

    return list_im, list_seg


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 100, 2),
                                    create_test_image(100, 100, 0, 1)])
def test_HistogramClipping(im_seg):
    im, _ = im_seg
    # Transform
    transform = HistogramClipping()
    # Apply Transform
    metadata = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})
    do_im, _ = transform(sample=im, metadata=metadata)
    # Check result has the same number of contrasts
    assert len(do_im) == len(im)
    # Check clipping
    min_percentile = transform.min_percentile
    max_percentile = transform.max_percentile
    for i, r in zip(im, do_im):
        assert isclose(np.min(r), np.percentile(i, min_percentile), rel_tol=1e-02)
        assert isclose(np.max(r), np.percentile(i, max_percentile), rel_tol=1e-02)


# @pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 100, 1),
#                                     create_test_image(100, 100, 0, 2)])
# def test_RandomShiftIntensity(im_seg):
#     im, _ = im_seg
#     # Transform
#     transform = RandomShiftIntensity(shift_range=[0., 10.], prob=0.9)
#
#     # Apply Do Transform
#     metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})
#     do_im, do_metadata = transform(sample=im, metadata=metadata_in)
#     # Check result has the same number of contrasts
#     assert len(do_im) == len(im)
#     # Check metadata update
#     assert all('offset' in m for m in do_metadata)
#     # Check shifting
#     for idx, i in enumerate(im):
#         assert isclose(np.max(do_im[idx] - i), do_metadata[idx]['offset'], rel_tol=1e-03)
#
#     # Apply Undo Transform
#     undo_im, undo_metadata = transform.undo_transform(sample=do_im, metadata=do_metadata)
#     # Check result has the same number of contrasts
#     assert len(undo_im) == len(im)
#     # Check undo
#     for idx, i in enumerate(im):
#         assert np.max(abs(undo_im[idx] - i)) <= 1e-03


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 100, 1),
                                    create_test_image(100, 100, 0, 2)])
def test_NumpyToTensor(im_seg):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform
    transform = NumpyToTensor()

    for im_cur in [im, seg]:
        # Numpy to Tensor
        do_im, do_metadata = transform(sample=im_cur, metadata=metadata_in)
        for idx, i in enumerate(do_im):
            assert torch.is_tensor(i)

        # Tensor to Numpy
        undo_im, undo_metadata = transform.undo_transform(sample=do_im, metadata=do_metadata)
        for idx, i in enumerate(undo_im):
            assert isinstance(i, np.ndarray)
            assert np.array_equal(i, im_cur[idx])
            assert i.dtype == im_cur[idx].dtype


def _test_Resample(im_seg, resample_transform, native_resolution, is_2D=False):
    im, seg = im_seg
    metadata_ = SampleMetadata({MetadataKW.ZOOMS: native_resolution,
                                MetadataKW.DATA_SHAPE: im[0].shape if len(im[0].shape) == 3 else list(im[0].shape) + [1],
                                MetadataKW.DATA_TYPE: 'im'
                                })
    metadata_in = [metadata_ for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Resample input data
    do_im, do_metadata = resample_transform(sample=im, metadata=metadata_in)
    # Undo Resample on input data
    undo_im, _ = resample_transform.undo_transform(sample=do_im, metadata=do_metadata)

    # Resampler for label data
    resample_transform.interpolation_order = 0
    metadata_ = SampleMetadata({MetadataKW.ZOOMS: native_resolution,
                                MetadataKW.DATA_SHAPE: seg[0].shape if len(seg[0].shape) == 3 else list(seg[0].shape) + [1],
                                MetadataKW.DATA_TYPE: 'gt'
                                })
    metadata_in = [metadata_ for _ in seg] if isinstance(seg, list) else SampleMetadata({})
    # Resample label data
    do_seg, do_metadata = resample_transform(sample=seg, metadata=metadata_in)
    # Undo Resample on label data
    undo_seg, _ = resample_transform.undo_transform(sample=do_seg, metadata=do_metadata)

    # Check data type and shape
    _check_dtype(im, [undo_im])
    _check_shape(im, [undo_im])
    _check_dtype(seg, [undo_seg])
    _check_shape(seg, [undo_seg])

    # Check data content and data shape between input data and undo
    for idx, i in enumerate(im):
        # Plot for debugging
        if DEBUGGING and is_2D:
            plot_transformed_sample(im[idx], undo_im[idx], ['raw', 'undo'])
            plot_transformed_sample(seg[idx], undo_seg[idx], ['raw', 'undo'])
        # Data consistency
        assert dice_score(undo_seg[idx], seg[idx]) > 0.8


@pytest.mark.parametrize('im_seg', [create_test_image(80, 100, 0, 2, rad_max=10)])
@pytest.mark.parametrize('resample_transform', [Resample(0.8, 1.0),
                                                Resample(1.0, 0.8)])
@pytest.mark.parametrize('native_resolution', [(0.9, 1.0),
                                               (1.0, 0.9)])
def test_Resample_2D(im_seg, resample_transform, native_resolution):
    _test_Resample(im_seg, resample_transform, native_resolution, is_2D=True)


@pytest.mark.parametrize('im_seg', [create_test_image(80, 100, 100, 1, rad_max=10)])
@pytest.mark.parametrize('resample_transform', [Resample(0.8, 1.0, 0.5),
                                                Resample(1.0, 0.8, 0.7)])
@pytest.mark.parametrize('native_resolution', [(0.9, 1.0, 0.8),
                                               (1.0, 0.9, 1.1)])
def test_Resample_3D(im_seg, resample_transform, native_resolution):
    _test_Resample(im_seg, resample_transform, native_resolution)


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 100, 1),
                                    create_test_image(100, 100, 0, 2)])
def test_NormalizeInstance(im_seg):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform on Numpy
    transform = NormalizeInstance()
    do_im, _ = transform(im.copy(), metadata_in)
    # Check normalization
    for i in do_im:
        assert abs(np.mean(i) - 0.0) <= 1e-2
        assert abs(np.std(i) - 1.0) <= 1e-2

    # Transform on Tensor
    tensor, metadata_tensor = NumpyToTensor()(im, metadata_in)
    do_tensor, _ = transform(tensor, metadata_tensor)
    # Check normalization
    assert abs(do_tensor.mean() - 0.0) <= 1e-2
    assert abs(do_tensor.std() - 1.0) <= 1e-2

    # Transform on Numpy - Uniform sample
    im = np.ones(im[0].shape)
    do_im, _ = transform(im.copy(), metadata_in)
    # Check mean-substraction
    assert abs(np.mean(do_im) - 0.0) <= 1e-2
    assert abs(np.std(do_im)) < 1e-5

    # Transform on Tensor - Uniform sample
    tensor, metadata_tensor = NumpyToTensor()(im, metadata_in)
    do_tensor, _ = transform(tensor, metadata_tensor)
    # Check mean-subtraction
    assert abs(do_tensor.mean() - 0.0) <= 1e-2
    assert abs(do_tensor.std()) < 1e-5


def _test_Crop(im_seg, crop_transform):
    im, seg = im_seg
    metadata_ = SampleMetadata({MetadataKW.DATA_SHAPE: im[0].shape, MetadataKW.CROP_PARAMS: {}})
    metadata_in = [metadata_ for _ in im] if isinstance(im, list) else {}
    if crop_transform.__class__.__name__ == "ROICrop":
        _, metadata_in = crop_transform(seg, metadata_in)
        for metadata in metadata_in:
            assert crop_transform.__class__.__name__ in metadata[MetadataKW.CROP_PARAMS]

    # Apply transform
    do_im, do_metadata = crop_transform(im, metadata_in)
    do_seg, do_seg_metadata = crop_transform(seg, metadata_in)
    crop_transfrom_size = crop_transform.size if not len(do_im[0].shape) == 2 else crop_transform.size[:2]

    # Loop and check
    for idx, i in enumerate(im):
        # Check data shape
        assert list(do_im[idx].shape) == crop_transfrom_size
        assert list(do_seg[idx].shape) == crop_transfrom_size
        # Check metadata
        assert do_metadata[idx][MetadataKW.CROP_PARAMS][crop_transform.__class__.__name__] == \
               do_seg_metadata[idx][MetadataKW.CROP_PARAMS][crop_transform.__class__.__name__]

    # Apply undo transform
    undo_im, _ = crop_transform.undo_transform(do_im, do_metadata)
    undo_seg, _ = crop_transform.undo_transform(do_seg, do_seg_metadata)

    # Check data type and shape
    _check_dtype(im, [undo_im])
    _check_shape(im, [undo_im])
    _check_dtype(seg, [undo_seg])
    _check_shape(seg, [undo_seg])

    # Loop and check
    for idx, i in enumerate(im):
        # Check data consistency
        fh, fw, fd, _, _, _ = do_metadata[idx][MetadataKW.CROP_PARAMS][crop_transform.__class__.__name__]
        th, tw, td = crop_transform.size
        if not td:
            assert np.array_equal(i[fh:fh + th, fw:fw + tw], undo_im[idx][fh:fh + th, fw:fw + tw])
            assert np.array_equal(seg[idx][fh:fh + th, fw:fw + tw], undo_seg[idx][fh:fh + th, fw:fw + tw])
            # Plot for debugging
            if DEBUGGING:
                plot_transformed_sample(seg[idx], undo_seg[idx], ['raw', 'undo'])
                plot_transformed_sample(i, undo_im[idx], ['raw', 'undo'])
        else:
            assert np.array_equal(i[fh:fh + th, fw:fw + tw, fd:fd + td],
                                  undo_im[idx][fh:fh + th, fw:fw + tw, fd:fd + td])
            assert np.array_equal(seg[idx][fh:fh + th, fw:fw + tw, fd:fd + td],
                                  undo_seg[idx][fh:fh + th, fw:fw + tw, fd:fd + td])


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 0, 2)])
@pytest.mark.parametrize('crop_transform', [CenterCrop([80, 60]),
                                            CenterCrop([60, 80]),
                                            ROICrop([80, 60]),
                                            ROICrop([60, 80])])
def test_Crop_2D(im_seg, crop_transform):
    _test_Crop(im_seg, crop_transform)


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 100, 1)])
@pytest.mark.parametrize('crop_transform', [CenterCrop([80, 60, 40]),
                                            CenterCrop([60, 80, 50]),
                                            ROICrop([80, 60, 40]),
                                            ROICrop([60, 80, 50])])
def test_Crop_3D(im_seg, crop_transform):
    _test_Crop(im_seg, crop_transform)


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 0, 1, rad_max=10),
                                    create_test_image(100, 100, 100, 1, rad_max=10)])
@pytest.mark.parametrize('transform', [RandomAffine(degrees=180),
                                       RandomAffine(degrees=(5, 180)),
                                       RandomAffine(translate=[0.1, 0.2, 0]),
                                       RandomAffine(scale=[0.03, 0.07, 0.0]),
                                       RandomAffine(translate=[0.1, 0.2, 0.05],
                                                    scale=[0.05, 0.05, 0],
                                                    degrees=5)])
def test_RandomAffine(im_seg, transform):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform on Numpy
    do_im, metadata_do = transform(im.copy(), metadata_in)
    do_seg, metadata_do = transform(seg.copy(), metadata_do)

    if DEBUGGING and len(im[0].shape) == 2:
        plot_transformed_sample(im[0], do_im[0], ['raw', 'do'])
        plot_transformed_sample(seg[0], do_seg[0], ['raw', 'do'])

    # Transform on Numpy
    undo_im, _ = transform.undo_transform(do_im, metadata_do)
    undo_seg, _ = transform.undo_transform(do_seg, metadata_do)

    if DEBUGGING and len(im[0].shape) == 2:
        # TODO: ERROR for image but not for seg.....
        plot_transformed_sample(im[0], undo_im[0], ['raw', 'undo'])
        plot_transformed_sample(seg[0], undo_seg[0], ['raw', 'undo'])

    # Check data type and shape
    _check_dtype(im, [do_im, undo_im])
    _check_shape(im, [do_im, undo_im])
    _check_dtype(seg, [undo_seg, do_seg])
    _check_shape(seg, [undo_seg, do_seg])

    # Loop and check
    for idx, i in enumerate(im):
        # Data consistency
        assert dice_score(undo_seg[idx], seg[idx]) > 0.85


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 0, 1, rad_max=10),
                                    create_test_image(100, 100, 100, 1, rad_max=10)])
@pytest.mark.parametrize('elastic_transform', [ElasticTransform(alpha_range=[150.0, 250.0],
                                                                sigma_range=[100 * 0.06, 100 * 0.09])])
def test_ElasticTransform(im_seg, elastic_transform):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform on Numpy
    do_im, metadata_do = elastic_transform(im.copy(), metadata_in)
    do_seg, metadata_do = elastic_transform(seg.copy(), metadata_do)

    if DEBUGGING and len(im[0].shape) == 2:
        plot_transformed_sample(im[0], do_im[0], ['raw', 'do'])
        plot_transformed_sample(seg[0], do_seg[0], ['raw', 'do'])

    _check_dtype(im, [do_im])
    _check_shape(im, [do_im])
    _check_dtype(seg, [do_seg])
    _check_shape(seg, [do_seg])


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 0, 1, rad_max=10),
                                    create_test_image(100, 100, 100, 1, rad_max=10)])
@pytest.mark.parametrize('dilate_transform', [DilateGT(dilation_factor=0.3)])
def test_DilateGT(im_seg, dilate_transform):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform on Numpy
    do_seg, metadata_do = dilate_transform(seg.copy(), metadata_in)

    if DEBUGGING and len(im[0].shape) == 2:
        plot_transformed_sample(seg[0], do_seg[0], ['raw', 'do'])

    # Check data shape and type
    _check_shape(ref=seg, list_mov=[do_seg])

    # Check data augmentation
    for idx, i in enumerate(seg):
        # data aug
        assert np.sum((do_seg[idx] > 0).astype(int)) >= np.sum(i)
        # same number of objects
        assert label((do_seg[idx] > 0).astype(int))[1] == label(i)[1]


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 0, 1, rad_max=10),
                                    create_test_image(100, 100, 100, 1, rad_max=10)])
@pytest.mark.parametrize('reverse_transform', [RandomReverse()])
def test_RandomReverse(im_seg, reverse_transform):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform on Numpy
    do_im, metadata_do = reverse_transform(im.copy(), metadata_in)
    do_seg, metadata_do = reverse_transform(seg.copy(), metadata_do)

    # Transform on Numpy
    undo_im, _ = reverse_transform.undo_transform(do_im, metadata_do)
    undo_seg, _ = reverse_transform.undo_transform(do_seg, metadata_do)

    if DEBUGGING and len(im[0].shape) == 2:
        plot_transformed_sample(seg[0], do_seg[0], ['raw', 'do'])
        plot_transformed_sample(seg[0], undo_seg[0], ['raw', 'undo'])

    _check_dtype(im, [do_im])
    _check_shape(im, [do_im])
    _check_dtype(seg, [do_seg])
    _check_shape(seg, [do_seg])


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 0, 1, rad_max=10),
                                    create_test_image(100, 100, 100, 1, rad_max=10)])
@pytest.mark.parametrize('noise_transform', [AdditiveGaussianNoise(mean=1., std=0.01)])
def test_AdditiveGaussianNoise(im_seg, noise_transform):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform on Numpy
    do_im, metadata_do = noise_transform(im.copy(), metadata_in)

    _check_dtype(im, [do_im])
    _check_shape(im, [do_im])

    if DEBUGGING and len(im[0].shape) == 2:
        plot_transformed_sample(im[0], do_im[0], ['raw', 'do'])


@pytest.mark.parametrize('im_seg', [create_test_image(100, 100, 0, 1, rad_max=10)])
@pytest.mark.parametrize('clahe', [Clahe(kernel_size=(8, 8))])
def test_Clahe(im_seg, clahe):
    im, seg = im_seg
    metadata_in = [SampleMetadata({}) for _ in im] if isinstance(im, list) else SampleMetadata({})

    # Transform on Numpy
    do_im, metadata_do = clahe(im.copy(), metadata_in)

    _check_dtype(im, [do_im])
    _check_shape(im, [do_im])

    if DEBUGGING and len(im[0].shape) == 2:
        plot_transformed_sample(im[0], do_im[0], ['raw', 'do'])


def _check_shape(ref, list_mov):
    # Loop and check
    for mov in list_mov:
        for idx, i in enumerate(ref):
            # Check data shape
            assert mov[idx].shape == i.shape


def _check_dtype(ref, list_mov):
    # Loop and check
    for mov in list_mov:
        for idx, i in enumerate(ref):
            # Check data type
            assert mov[idx].dtype == i.dtype
