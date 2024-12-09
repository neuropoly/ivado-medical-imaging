from ivadomed import visualize as imed_visualize
import numpy as np
import torch
from testing.unit_tests.t_utils import create_tmp_dir,  __data_testing_dir__, __tmp_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path

def setup_function():
    create_tmp_dir()


def test_save_rgb(download_data_testing_test_files):
    # Create image with shape HxWxDxC
    image = [[[[0 for i in range(10)] for i in range(10)] for i in range(4)] for i in range(3)]
    for i in range(3):
        image[0][2][5][i] = 1
        image[1][2][5][i+3] = 1
        image[2][2][5][i + 6] = 1
    image_n = np.array(image)
    imed_visualize.save_color_labels(
        gt_data=image_n,
        binarize=False,
        gt_filename=str(Path(
            __data_testing_dir__,
            "rgb_test_file.nii.gz")),
        output_filename=str(Path(__tmp_dir__, "rgb_test.nii.gz")),
        slice_axis=0
    )


def test_rgb_conversion():
    # Create image with shape HxWxn_classxbatch_size
    image = [[[[0 for i in range(10)] for i in range(10)] for i in range(3)] for i in range(3)]
    for i in range(3):
        image[0][2][5][i] = 1
        image[1][2][5][i + 3] = 1
        image[2][2][5][i + 6] = 1
    image_n = np.array(image)
    tensor_multi = torch.tensor(image_n)
    imed_visualize.convert_labels_to_RGB(tensor_multi)


def teardown_function():
    remove_tmp_dir()
