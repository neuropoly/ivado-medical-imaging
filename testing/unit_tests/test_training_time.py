import time
import numpy as np
import pytest
import torch.backends.cudnn as cudnn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from loguru import logger

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed import losses as imed_losses
from ivadomed import models as imed_models
from ivadomed import utils as imed_utils
from ivadomed.loader import utils as imed_loader_utils, loader as imed_loader
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir

cudnn.benchmark = True

GPU_ID = 0
BATCH_SIZE = 8
N_EPOCHS = 2
INIT_LR = 0.01
MODEL_DEFAULT = {
    "dropout_rate": 0.3,
    "bn_momentum": 0.1,
    "depth": 3
}


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize('target_lst', [["_lesion-manual"]])
@pytest.mark.parametrize('train_lst, config', [
    (['sub-unf01_T2w.nii.gz'], {
        "transforms_params": {"Resample": {"wspace": 0.75, "hspace": 0.75},
                              "ROICrop": {"size": [48, 48]},
                              "NumpyToTensor": {}},
        "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": 10},
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "multichannel": False,
        "model_params": {"name": "Unet"},
    }),
    (['sub-unf01_T1w.nii.gz', 'sub-unf01_T2w.nii.gz'], {
        "transforms_params": {"Resample": {"wspace": 0.75, "hspace": 0.75},
                              "ROICrop": {"size": [48, 48]},
                              "NumpyToTensor": {}},
        "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": 10},
        "contrast_params": {"contrast_lst": ['T1w', 'T2w'], "balance": {}},
        "multichannel": True,
        "model_params": {"name": "Unet"},
    }),
    (['sub-unf01_T1w.nii.gz', 'sub-unf01_T2w.nii.gz'], {
        "transforms_params": {"CenterCrop": {"size": [96, 96, 16]},
                              "NumpyToTensor": {}},
        "roi_params": {"suffix": None, "slice_filter_roi": 0},
        "contrast_params": {"contrast_lst": ['T1w', 'T2w'], "balance": {}},
        "multichannel": False,
        "model_params": {"name": "Modified3DUNet", "length_3D": [96, 96, 16], "n_filters": 8, "stride_3D": [96, 96, 16],
                         "attention": True},
    }),
    (['sub-unf01_T1w.nii.gz', 'sub-unf01_T2w.nii.gz'], {
        "transforms_params": {"CenterCrop": {"size": [96, 96, 16]},
                              "NumpyToTensor": {}},
        "roi_params": {"suffix": None, "slice_filter_roi": 0},
        "contrast_params": {"contrast_lst": ['T1w', 'T2w'], "balance": {}},
        "multichannel": False,
        "model_params": {"name": "Modified3DUNet", "length_3D": [96, 96, 16], "n_filters": 8, "stride_3D": [96, 96, 16],
                         "attention": False},
    }),
    (['sub-unf01_T1w.nii.gz', 'sub-unf01_T2w.nii.gz'], {
        "transforms_params": {"CenterCrop": {"size": [96, 96, 16]},
                              "NumpyToTensor": {}},
        "roi_params": {"suffix": None, "slice_filter_roi": 0},
        "contrast_params": {"contrast_lst": ['T1w', 'T2w'], "balance": {}},
        "multichannel": False,
        "model_params": {"name": "Modified3DUNet", "length_3D": [96, 96, 16], "n_filters": 8, "stride_3D": [96, 96, 16],
                         "attention": False, "metadata": "contrasts", "film_layers": [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                         "n_metadata": 2},
    }),
    (['sub-unf01_T2w.nii.gz'], {
        "transforms_params": {"CenterCrop": {"size": [96, 96, 16]},
                              "NumpyToTensor": {}},
        "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": 10},
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "multichannel": False,
        "model_params": {"name": "Unet", 'is_2d': False, "length_3D": [96, 96, 16], "n_filters": 8,
                         "stride_3D": [96, 96, 16]},
    })
])
def test_unet_time(download_data_testing_test_files, train_lst, target_lst, config):
    cuda_available, device = imed_utils.define_device(GPU_ID)

    loader_params = {
        "data_list": train_lst,
        "dataset_type": "training",
        "requires_undo": False,
        "path_data": [__data_testing_dir__],
        "target_suffix": target_lst,
        "extensions": [".nii.gz"],
        "slice_filter_params": {"filter_empty_mask": False, "filter_empty_input": True},
        "patch_filter_params": {"filter_empty_mask": False, "filter_empty_input": False},
        "slice_axis": "axial"
    }
    # Update loader_params with config
    loader_params.update(config)
    # Get Training dataset
    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)
    ds_train = imed_loader.load_dataset(bids_df, **loader_params)

    # Loader
    train_loader = DataLoader(ds_train,
                              batch_size=1 if config["model_params"]["name"] == "Modified3DUNet" else BATCH_SIZE,
                              shuffle=True, pin_memory=True,
                              collate_fn=imed_loader_utils.imed_collate,
                              num_workers=1)

    # MODEL
    model_params = loader_params["model_params"]
    model_params.update(MODEL_DEFAULT)
    # Get in_channel from contrast_lst
    if loader_params["multichannel"]:
        model_params["in_channel"] = len(loader_params["contrast_params"]["contrast_lst"])
    else:
        model_params["in_channel"] = 1
    # Get out_channel from target_suffix
    model_params["out_channel"] = len(loader_params["target_suffix"])
    model_class = getattr(imed_models, model_params["name"])
    model = model_class(**model_params)

    logger.debug(f"Training {model_params['name']}")
    if cuda_available:
        model.cuda()

    step_scheduler_batch = False
    # TODO: Add optim in pytest
    optimizer = optim.Adam(model.parameters(), lr=INIT_LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, N_EPOCHS)

    # TODO: add to pytest
    loss_fct = imed_losses.DiceLoss()

    load_lst, pred_lst, opt_lst, schedule_lst, init_lst, gen_lst = [], [], [], [], [], []
    for epoch in tqdm(range(1, N_EPOCHS + 1), desc="Training"):
        start_time = time.time()

        start_init = time.time()

        model.train()

        tot_init = time.time() - start_init
        init_lst.append(tot_init)

        num_steps = 0
        start_gen = 0
        for i, batch in enumerate(train_loader):
            if i > 0:
                tot_gen = time.time() - start_gen
                gen_lst.append(tot_gen)

            start_load = time.time()
            input_samples = imed_utils.cuda(batch["input"], cuda_available)
            gt_samples = imed_utils.cuda(batch["gt"], cuda_available, non_blocking=True)

            tot_load = time.time() - start_load
            load_lst.append(tot_load)

            start_pred = time.time()

            if 'film_layers' in model_params:
                preds = model(input_samples, [[0, 1]])
            else:
                preds = model(input_samples)
            tot_pred = time.time() - start_pred
            pred_lst.append(tot_pred)

            start_opt = time.time()
            loss = loss_fct(preds, gt_samples)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step_scheduler_batch:
                scheduler.step()

            num_steps += 1
            tot_opt = time.time() - start_opt
            opt_lst.append(tot_opt)

            start_gen = time.time()

        start_schedule = time.time()
        if not step_scheduler_batch:
            scheduler.step()
        tot_schedule = time.time() - start_schedule
        schedule_lst.append(tot_schedule)

        end_time = time.time()
        total_time = end_time - start_time
        tqdm.write("Epoch {} took {:.2f} seconds.".format(epoch, total_time))

    logger.info(f"Mean SD init {np.mean(init_lst)} -- {np.std(init_lst)}")
    logger.info(f"Mean SD load {np.mean(load_lst)} -- {np.std(load_lst)}")
    logger.info(f"Mean SD pred {np.mean(pred_lst)} -- {np.std(pred_lst)}")
    logger.info(f"Mean SDopt {np.mean(opt_lst)} --  {np.std(opt_lst)}")
    logger.info(f"Mean SD gen {np.mean(gen_lst)} -- {np.std(gen_lst)}")
    logger.info(f"Mean SD scheduler {np.mean(schedule_lst)} -- {np.std(schedule_lst)}")


def teardown_function():
    remove_tmp_dir()
