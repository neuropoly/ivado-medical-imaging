import pytest
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from pathlib import Path

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed import metrics as imed_metrics
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils
from ivadomed import testing as imed_testing
from ivadomed import models as imed_models
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
import logging
from testing.unit_tests.t_utils import create_tmp_dir, __data_testing_dir__, __tmp_dir__, download_data_testing_test_files, path_repo_root
from testing.common_testing_util import remove_tmp_dir
logger = logging.getLogger(__name__)

cudnn.benchmark = True

GPU_ID = 0
BATCH_SIZE = 8
DROPOUT = 0.4
BN = 0.1
SLICE_AXIS = "axial"
__output_dir__ = Path(__tmp_dir__, "output_inference")


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('transforms_dict', [{
        "Resample": {
                "wspace": 0.75,
                "hspace": 0.75
            },
        "CenterCrop": {
                "size": [48, 48]
            },
        "NormalizeInstance": {"applied_to": ["im"]}
    }])
@pytest.mark.parametrize('test_lst', [['sub-unf01_T2w.nii.gz']])
@pytest.mark.parametrize('target_lst', [["_lesion-manual"], ["_seg-manual"]])
@pytest.mark.parametrize('roi_params', [{"suffix": "_seg-manual", "slice_filter_roi": 10}])
@pytest.mark.parametrize('testing_params', [{
    "binarize_prediction": 0.5,
    "uncertainty": {
        "applied": False,
        "epistemic": False,
        "aleatoric": False,
        "n_it": 0
    }}])
def test_inference(download_data_testing_test_files, transforms_dict, test_lst, target_lst, roi_params, testing_params):
    cuda_available, device = imed_utils.define_device(GPU_ID)

    model_params = {"name": "Unet", "is_2d": True}
    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": test_lst,
        "dataset_type": "testing",
        "requires_undo": True,
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "path_data": [__data_testing_dir__],
        "target_suffix": target_lst,
        "extensions": [".nii.gz"],
        "roi_params": roi_params,
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
        "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "slice_axis": SLICE_AXIS,
        "multichannel": False
    }
    loader_params.update({"model_params": model_params})

    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)

    # Get Testing dataset
    ds_test = imed_loader.load_dataset(bids_df, **loader_params)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # Undo transform
    val_undo_transform = imed_transforms.UndoCompose(imed_transforms.Compose(transforms_dict))

    # Update testing_params
    testing_params.update({
        "slice_axis": loader_params["slice_axis"],
        "target_suffix": loader_params["target_suffix"],
        "undo_transforms": val_undo_transform
    })

    # Model
    model = imed_models.Unet()

    if cuda_available:
        model.cuda()
    model.eval()

    metric_fns = [imed_metrics.dice_score,
                  imed_metrics.hausdorff_score,
                  imed_metrics.precision_score,
                  imed_metrics.recall_score,
                  imed_metrics.specificity_score,
                  imed_metrics.intersection_over_union,
                  imed_metrics.accuracy_score]

    metric_mgr = imed_metrics.MetricManager(metric_fns)

    if not __output_dir__.is_dir():
        __output_dir__.mkdir(parents=True, exist_ok=True)

    preds_npy, gt_npy = imed_testing.run_inference(test_loader=test_loader,
                                                   model=model,
                                                   model_params=model_params,
                                                   testing_params=testing_params,
                                                   ofolder=str(__output_dir__),
                                                   cuda_available=cuda_available)

    metric_mgr(preds_npy, gt_npy)
    metrics_dict = metric_mgr.get_results()
    metric_mgr.reset()
    logger.debug(metrics_dict)


@pytest.mark.parametrize('transforms_dict', [{
        "CenterCrop": {
                "size": [128, 128]
            },
        "NormalizeInstance": {"applied_to": ["im"]}
    }])
@pytest.mark.parametrize('test_lst',
    [['sub-rat3_ses-01_sample-data9_SEM.png', 'sub-rat3_ses-02_sample-data10_SEM.png']])
@pytest.mark.parametrize('target_lst', [["_seg-axon-manual", "_seg-myelin-manual"]])
@pytest.mark.parametrize('roi_params', [{"suffix": None, "slice_filter_roi": None}])
@pytest.mark.parametrize('testing_params', [{
    "binarize_maxpooling": {},
    "uncertainty": {
        "applied": False,
        "epistemic": False,
        "aleatoric": False,
        "n_it": 0
    }}])
def test_inference_2d_microscopy(download_data_testing_test_files, transforms_dict, test_lst, target_lst, roi_params,
        testing_params):
    """
    This test checks if the number of NifTI predictions equals the number of test subjects on 2d microscopy data.
    Used to catch a bug where the last slice of the last volume wasn't appended to the prediction
    (see: https://github.com/ivadomed/ivadomed/issues/823)
    Also tests the conversions to PNG predictions when source files are not Nifti and checks if the number of PNG
    predictions is 2x the number of test subjects (2-class model, outputs 1 PNG per class per subject).
    """
    cuda_available, device = imed_utils.define_device(GPU_ID)

    model_params = {"name": "Unet", "is_2d": True, "out_channel": 3}
    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": test_lst,
        "dataset_type": "testing",
        "requires_undo": True,
        "contrast_params": {"contrast_lst": ['SEM'], "balance": {}},
        "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
        "target_suffix": target_lst,
        "extensions": [".png"],
        "roi_params": roi_params,
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
        "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "slice_axis": SLICE_AXIS,
        "multichannel": False
    }
    loader_params.update({"model_params": model_params})

    bids_df = BidsDataframe(loader_params, str(__tmp_dir__), derivatives=True)

    # Get Testing dataset
    ds_test = imed_loader.load_dataset(bids_df, **loader_params)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # Undo transform
    val_undo_transform = imed_transforms.UndoCompose(imed_transforms.Compose(transforms_dict))

    # Update testing_params
    testing_params.update({
        "slice_axis": loader_params["slice_axis"],
        "target_suffix": loader_params["target_suffix"],
        "undo_transforms": val_undo_transform
    })

    # Model
    model = imed_models.Unet(out_channel=model_params['out_channel'])

    if cuda_available:
        model.cuda()
    model.eval()

    if not __output_dir__.is_dir():
        __output_dir__.mkdir(parents=True, exist_ok=True)

    preds_npy, gt_npy = imed_testing.run_inference(test_loader=test_loader,
                                                   model=model,
                                                   model_params=model_params,
                                                   testing_params=testing_params,
                                                   ofolder=str(__output_dir__),
                                                   cuda_available=cuda_available)

    assert len([x for x in __output_dir__.iterdir() if x.name.endswith(".nii.gz")]) == len(test_lst) + 1
    assert len([x for x in __output_dir__.iterdir() if x.name.endswith(".png")]) == 2*len(test_lst)


@pytest.mark.parametrize('transforms_dict', [{
        "CenterCrop": {
                "size": [128, 128]
            },
        "NormalizeInstance": {"applied_to": ["im"]}
    }])
@pytest.mark.parametrize('test_lst',
    [['sub-rat3_ses-01_sample-data9_SEM.png', 'sub-rat3_ses-02_sample-data10_SEM.png']])
@pytest.mark.parametrize('target_lst', [["_seg-axon_manual", "_seg-myelin_manual"]])
@pytest.mark.parametrize('roi_params', [{"suffix": None, "slice_filter_roi": None}])
@pytest.mark.parametrize('testing_params', [{
    "binarize_maxpooling": {},
    "uncertainty": {
        "applied": False,
        "epistemic": False,
        "aleatoric": False,
        "n_it": 0
    }}])
def test_inference_target_suffix(download_data_testing_test_files, transforms_dict, test_lst, target_lst, roi_params,
        testing_params):
    """
    This test checks if the filename(s) of the prediction(s) saved as NifTI file(s) in the pred_masks
    dir conform to the target_suffix or not. Thus, independent of underscore(s) in the target_suffix. As a result,
    _seg-axon-manual or _seg-axon_manual should yield the same filename(s).
    (c.f: https://github.com/ivadomed/ivadomed/issues/1135)
    """
    cuda_available, device = imed_utils.define_device(GPU_ID)

    model_params = {"name": "Unet", "is_2d": True, "out_channel": 3}
    loader_params = {
        "transforms_params": transforms_dict,
        "data_list": test_lst,
        "dataset_type": "testing",
        "requires_undo": True,
        "contrast_params": {"contrast_lst": ['SEM'], "balance": {}},
        "path_data": [str(Path(__data_testing_dir__, "microscopy_png"))],
        "target_suffix": target_lst,
        "extensions": [".png"],
        "roi_params": roi_params,
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
        "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "slice_axis": SLICE_AXIS,
        "multichannel": False
    }
    loader_params.update({"model_params": model_params})

    # restructuring the dataset 
    gt_path = f'{loader_params["path_data"][0]}/derivatives/labels/'
    for file_path in Path(gt_path).rglob('*.png'):
      src_filename = file_path.resolve()
      dst_filename = '_'.join(str(src_filename).rsplit('-', 1))
      src_filename.rename(Path(dst_filename))

    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)

    ds_test = imed_loader.load_dataset(bids_df, **loader_params)
    test_loader = DataLoader(ds_test, batch_size=BATCH_SIZE,
                             shuffle=False, pin_memory=True,
                             collate_fn=imed_loader_utils.imed_collate,
                             num_workers=0)

    # Undo transform
    val_undo_transform = imed_transforms.UndoCompose(imed_transforms.Compose(transforms_dict))

    # Update testing_params
    testing_params.update({
        "slice_axis": loader_params["slice_axis"],
        "target_suffix": loader_params["target_suffix"],
        "undo_transforms": val_undo_transform
    })

    # Model
    model = imed_models.Unet(out_channel=model_params['out_channel'])

    if cuda_available:
        model.cuda()
    model.eval()

    if not __output_dir__.is_dir():
        __output_dir__.mkdir(parents=True, exist_ok=True)

    preds_npy, gt_npy = imed_testing.run_inference(test_loader=test_loader,
                                                   model=model,
                                                   model_params=model_params,
                                                   testing_params=testing_params,
                                                   ofolder=str(__output_dir__),
                                                   cuda_available=cuda_available)

    for x in __output_dir__.iterdir():
      if x.name.endswith('_pred.nii.gz'):
        assert x.name.rsplit('_', 1)[0].endswith(loader_params['contrast_params']['contrast_lst'][-1]), ( 
            'Incompatible filename(s) of the prediction(s) saved as NifTI file(s)!'
        )

def teardown_function():
    remove_tmp_dir()
