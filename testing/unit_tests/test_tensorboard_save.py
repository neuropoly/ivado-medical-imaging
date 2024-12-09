import ivadomed.visualize as imed_visualize
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import torch
import io
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import ivadomed.maths as imed_math
from PIL import Image
from testing.unit_tests.t_utils import create_tmp_dir,  __tmp_dir__
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path


def setup_function():
    create_tmp_dir()


def test_tensorboard_save():
    inp = torch.tensor(np.zeros((1, 1, 15, 15)))
    gt = torch.tensor(np.zeros((1, 1, 15, 15)))
    pred = torch.tensor(np.zeros((1, 1, 15, 15)))
    dpath = Path(__tmp_dir__, "test_tensorboard_save")
    dpath.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(dpath))
    imed_visualize.save_img(writer, 1, "Training", inp, pred, gt)
    writer.flush()

    summary_iterators = [EventAccumulator(str(dname)).Reload() for dname in dpath.iterdir()]
    for i in range(len(summary_iterators)):
        if summary_iterators[i].Tags()['images'] == ['Training/Input', 'Training/Predictions', 'Training/Ground Truth']:
            input_retrieve = np.array(Image.open(io.BytesIO(
                summary_iterators[i].Images('Training/Input')[0].encoded_image_string)))
            pred_retrieve = np.array(Image.open(io.BytesIO(
                summary_iterators[i].Images('Training/Predictions')[0].encoded_image_string)))
            gt_retrieve = np.array(Image.open(io.BytesIO(
                summary_iterators[i].Images('Training/Ground Truth')[0].encoded_image_string)))

    assert np.allclose(imed_math.rescale_values_array(input_retrieve[:, :, 0], 0, 1), inp[0, 0, :, :])
    assert np.allclose(imed_math.rescale_values_array(pred_retrieve[:, :, 0], 0, 1), pred[0, 0, :, :])
    assert np.allclose(imed_math.rescale_values_array(gt_retrieve[:, :, 0], 0, 1), gt[0, 0, :, :])


def teardown_function():
    remove_tmp_dir()
