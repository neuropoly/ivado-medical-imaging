import logging
from testing.functional_tests.t_utils import create_tmp_dir, __data_testing_dir__, download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed.scripts import prepare_dataset_vertebral_labeling
from pathlib import Path
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_prepare_dataset_vertebral_labeling(download_functional_test_files):
    prepare_dataset_vertebral_labeling.main(args=['--path', __data_testing_dir__,
                                                  '--suffix', '_T2w',
                                                  '--aim', '3'])
    assert Path(
        __data_testing_dir__, "derivatives/labels/sub-unf01/anat/sub-unf01_T2w_mid_heatmap3.nii.gz").exists()
    assert Path(
        __data_testing_dir__, "derivatives/labels/sub-unf02/anat/sub-unf02_T2w_mid_heatmap3.nii.gz").exists()
    assert Path(
        __data_testing_dir__, "derivatives/labels/sub-unf03/anat/sub-unf03_T2w_mid_heatmap3.nii.gz").exists()


def teardown_function():
    remove_tmp_dir()
