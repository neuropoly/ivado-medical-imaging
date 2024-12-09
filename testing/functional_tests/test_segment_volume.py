import json
import shutil
import nibabel as nib
import numpy as np
import pytest
import torch
from ivadomed import models as imed_models
from ivadomed import inference as imed_inference
from testing.functional_tests.t_utils import create_tmp_dir, __data_testing_dir__, __tmp_dir__, download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path

BATCH_SIZE = 1
DROPOUT = 0.3
BN = 0.1
SLICE_AXIS = 2
LENGTH_3D = [96, 96, 16]
LENGTH_2D = [128, 128]


def setup_function():
    create_tmp_dir()


PATH_MODEL = Path(__tmp_dir__, "model_test")
PATH_IMAGE = Path(__data_testing_dir__, "sub-unf01", "anat", "sub-unf01_T1w.nii.gz")
PATH_ROI = Path(__data_testing_dir__, "derivatives", "labels", "sub-unf01", "anat",
                        "sub-unf01_T1w_seg-manual.nii.gz")


def test_segment_volume_2d_NumpyToTensor_retrocompatibility(download_functional_test_files):
    model = imed_models.Unet(in_channel=1,
                             out_channel=1,
                             depth=2,
                             dropout_rate=DROPOUT,
                             bn_momentum=BN)

    if not PATH_MODEL.exists():
        PATH_MODEL.mkdir(parents=True, exist_ok=True)

    torch.save(model, Path(PATH_MODEL, "model_test.pt"))
    config = {
        "loader_parameters": {
            "slice_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "roi_params": {
                "suffix": "_seg-manual",
                "slice_filter_roi": 10
            },
            "slice_axis": "axial"
        },
        "transformation": {
            "Resample": {"wspace": 0.75, "hspace": 0.75},
            "ROICrop": {"size": [48, 48]},
            "RandomTranslation": {
                "translate": [0.03, 0.03],
                "applied_to": ["im", "gt"],
                "dataset_type": ["training"]
            },
            "NumpyToTensor": {},
            "NormalizeInstance": {"applied_to": ["im"]}
        },
        "postprocessing": {},
        "training_parameters": {
            "batch_size": BATCH_SIZE
        }
    }

    PATH_CONFIG = Path(PATH_MODEL, 'model_test.json')
    with open(PATH_CONFIG, 'w') as fp:
        json.dump(config, fp)

    nib_lst, _ = imed_inference.segment_volume(str(PATH_MODEL), [str(PATH_IMAGE)], options={'fname_prior': str(PATH_ROI)})
    nib_img = nib_lst[0]
    assert np.squeeze(nib_img.get_fdata()).shape == nib.load(PATH_IMAGE).shape
    assert (nib_img.dataobj.max() <= 1.0) and (nib_img.dataobj.min() >= 0.0)
    assert nib_img.dataobj.dtype == 'float32'

    shutil.rmtree(PATH_MODEL)


def test_segment_volume_2d(download_functional_test_files):
    model = imed_models.Unet(in_channel=1,
                             out_channel=1,
                             depth=2,
                             dropout_rate=DROPOUT,
                             bn_momentum=BN)

    if not PATH_MODEL.exists():
        PATH_MODEL.mkdir(parents=True, exist_ok=True)

    torch.save(model, Path(PATH_MODEL, "model_test.pt"))
    config = {
        "loader_parameters": {
            "slice_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "roi_params": {
                "suffix": "_seg-manual",
                "slice_filter_roi": 10
            },
            "slice_axis": "axial"
        },
        "transformation": {
            "Resample": {"wspace": 0.75, "hspace": 0.75},
            "ROICrop": {"size": [48, 48]},
            "RandomTranslation": {
                "translate": [0.03, 0.03],
                "applied_to": ["im", "gt"],
                "dataset_type": ["training"]
            },
            "NormalizeInstance": {"applied_to": ["im"]}
        },
        "postprocessing": {},
        "training_parameters": {
            "batch_size": BATCH_SIZE
        }
    }

    PATH_CONFIG = Path(PATH_MODEL, 'model_test.json')
    with PATH_CONFIG.open(mode='w') as fp:
        json.dump(config, fp)

    nib_lst, _ = imed_inference.segment_volume(str(PATH_MODEL), [str(PATH_IMAGE)], options={'fname_prior': str(PATH_ROI)})
    nib_img = nib_lst[0]
    assert np.squeeze(nib_img.get_fdata()).shape == nib.load(PATH_IMAGE).shape
    assert (nib_img.dataobj.max() <= 1.0) and (nib_img.dataobj.min() >= 0.0)
    assert nib_img.dataobj.dtype == 'float32'

    shutil.rmtree(PATH_MODEL)


def test_segment_volume_2d_no_prepro_transform(download_functional_test_files):
    model = imed_models.Unet(in_channel=1,
                             out_channel=1,
                             depth=2,
                             dropout_rate=DROPOUT,
                             bn_momentum=BN)

    if not PATH_MODEL.exists():
        PATH_MODEL.mkdir()

    torch.save(model, Path(PATH_MODEL, "model_test.pt"))
    config = {
        "loader_parameters": {
            "slice_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "roi_params": {
                "suffix": None,
                "slice_filter_roi": None
            },
            "slice_axis": "axial"
        },
        "transformation": {
            "NormalizeInstance": {"applied_to": ["im"]}
        },
        "postprocessing": {},
        "training_parameters": {
            "batch_size": BATCH_SIZE
        }
    }

    PATH_CONFIG = Path(PATH_MODEL, 'model_test.json')
    with PATH_CONFIG.open(mode='w') as fp:
        json.dump(config, fp)

    nib_lst, _ = imed_inference.segment_volume(str(PATH_MODEL), [str(PATH_IMAGE)])
    nib_img = nib_lst[0]
    assert np.squeeze(nib_img.get_fdata()).shape == nib.load(PATH_IMAGE).shape
    assert (nib_img.dataobj.max() <= 1.0) and (nib_img.dataobj.min() >= 0.0)
    assert nib_img.dataobj.dtype == 'float32'

    shutil.rmtree(PATH_MODEL)


@pytest.mark.parametrize("center_crop", [[200, 200]])
def test_segment_volume_2d_with_patches(download_functional_test_files, center_crop):
    model = imed_models.Unet(in_channel=1,
                             out_channel=1,
                             depth=2,
                             dropout_rate=DROPOUT,
                             bn_momentum=BN)

    if not PATH_MODEL.exists():
        PATH_MODEL.mkdir(parents=True, exist_ok=True)

    torch.save(model, Path(PATH_MODEL, "model_test.pt"))
    config = {
        "loader_parameters": {
            "slice_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "patch_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "roi_params": {
                "suffix": None,
                "slice_filter_roi": None
            },
            "slice_axis": "axial"
        },
        "default_model": {
            "length_2D": LENGTH_2D,
            "stride_2D": LENGTH_2D
        },
        "transformation": {
            "Resample": {"wspace": 0.75, "hspace": 0.75},
            "CenterCrop": {"size": center_crop},
            "RandomTranslation": {
                "translate": [0.03, 0.03],
                "applied_to": ["im", "gt"],
                "dataset_type": ["training"]
            },
            "NormalizeInstance": {"applied_to": ["im"]}
        },
        "postprocessing": {},
        "training_parameters": {
            "batch_size": BATCH_SIZE
        }
    }

    PATH_CONFIG = Path(PATH_MODEL, 'model_test.json')
    with PATH_CONFIG.open(mode='w') as fp:
        json.dump(config, fp)

    nib_lst, _ = imed_inference.segment_volume(str(PATH_MODEL), [str(PATH_IMAGE)])
    nib_img = nib_lst[0]
    assert np.squeeze(nib_img.get_fdata()).shape == nib.load(PATH_IMAGE).shape
    assert (nib_img.dataobj.max() <= 1.0) and (nib_img.dataobj.min() >= 0.0)
    assert nib_img.dataobj.dtype == 'float32'

    shutil.rmtree(PATH_MODEL)


@pytest.mark.parametrize("center_crop", [[200, 200]])
def test_segment_volume_2d_without_patches(download_functional_test_files, center_crop):
    model = imed_models.Unet(in_channel=1,
                             out_channel=1,
                             depth=2,
                             dropout_rate=DROPOUT,
                             bn_momentum=BN)

    if not PATH_MODEL.exists():
        PATH_MODEL.mkdir(parents=True, exist_ok=True)

    torch.save(model, Path(PATH_MODEL, "model_test.pt"))
    config = {
        "loader_parameters": {
            "slice_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "patch_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "roi_params": {
                "suffix": None,
                "slice_filter_roi": None
            },
            "slice_axis": "axial"
        },
        "default_model": {
            "length_2D": LENGTH_2D,
            "stride_2D": LENGTH_2D
        },
        "transformation": {
            "Resample": {"wspace": 0.75, "hspace": 0.75},
            "CenterCrop": {"size": center_crop},
            "RandomTranslation": {
                "translate": [0.03, 0.03],
                "applied_to": ["im", "gt"],
                "dataset_type": ["training"]
            },
            "NormalizeInstance": {"applied_to": ["im"]}
        },
        "postprocessing": {},
        "training_parameters": {
            "batch_size": BATCH_SIZE
        }
    }

    PATH_CONFIG = Path(PATH_MODEL, 'model_test.json')
    with PATH_CONFIG.open(mode='w') as fp:
        json.dump(config, fp)

    options = {}
    options['no_patch'] = True

    nib_lst, _ = imed_inference.segment_volume(str(PATH_MODEL), [str(PATH_IMAGE)], options=options)
    nib_img = nib_lst[0]
    assert np.squeeze(nib_img.get_fdata()).shape == nib.load(PATH_IMAGE).shape
    assert (nib_img.dataobj.max() <= 1.0) and (nib_img.dataobj.min() >= 0.0)
    assert nib_img.dataobj.dtype == 'float32'

    shutil.rmtree(PATH_MODEL)


@pytest.mark.parametrize("center_crop", [[192, 192, 16]])
def test_segment_volume_3d(download_functional_test_files, center_crop):
    model = imed_models.Modified3DUNet(in_channel=1,
                                       out_channel=1,
                                       base_n_filter=1)

    if not PATH_MODEL.exists():
        PATH_MODEL.mkdir(parents=True, exist_ok=True)

    torch.save(model, Path(PATH_MODEL, "model_test.pt"))
    config = {
        "Modified3DUNet": {
            "applied": True,
            "length_3D": LENGTH_3D,
            "stride_3D": LENGTH_3D,
            "attention": False
        },
        "loader_parameters": {
            "slice_filter_params": {
                "filter_empty_mask": False,
                "filter_empty_input": False
            },
            "roi_params": {
                "suffix": None,
                "slice_filter_roi": None
            },
            "slice_axis": "sagittal"
        },
        "transformation": {
            "Resample":
                {
                    "wspace": 1,
                    "hspace": 1,
                    "dspace": 2
                },
            "CenterCrop": {
                "size": center_crop
                },
            "RandomTranslation": {
                "translate": [0.03, 0.03],
                "applied_to": ["im", "gt"],
                "dataset_type": ["training"]
                },
            "NormalizeInstance": {"applied_to": ["im"]}
        },
        "postprocessing": {},
        "training_parameters": {
            "batch_size": BATCH_SIZE
        }
    }

    PATH_CONFIG = Path(PATH_MODEL, 'model_test.json')
    with PATH_CONFIG.open(mode='w') as fp:
        json.dump(config, fp)

    nib_lst, _ = imed_inference.segment_volume(str(PATH_MODEL), [str(PATH_IMAGE)])
    nib_img = nib_lst[0]
    assert np.squeeze(nib_img.get_fdata()).shape == nib.load(PATH_IMAGE).shape
    assert (nib_img.dataobj.max() <= 1.0) and (nib_img.dataobj.min() >= 0.0)
    assert nib_img.dataobj.dtype == 'float32'

    shutil.rmtree(PATH_MODEL)


def teardown_function():
    remove_tmp_dir()

