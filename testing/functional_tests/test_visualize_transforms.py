import logging
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed.scripts import visualize_transforms
from pathlib import Path

logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_visualize_transforms_n_1(download_functional_test_files):
    __data_testing_dir__ = Path(__tmp_dir__, "data_functional_testing")
    __input_file__ = Path(__data_testing_dir__, 'sub-unf01/anat/sub-unf01_T1w.nii.gz')
    __output_dir__ = Path(__tmp_dir__, "output_visualize_transforms_n_1")
    __config_file__ = Path(__data_testing_dir__, "model_config.json")
    __label_file__ = Path(__data_testing_dir__,
                                  'derivatives/labels/sub-test001/anat/sub-unf01_T1w_seg-manual.nii.gz')
    visualize_transforms.main(args=['--input', str(__input_file__),
                                    '--output', str(__output_dir__),
                                    '--config', str(__config_file__),
                                    '-r', str(__label_file__)])
    assert __output_dir__.exists()
    output_files = [f.name for f in __output_dir__.iterdir()]
    assert len(output_files) == 5
    for output_file in output_files:
        assert "Resample" in output_file
        assert "slice" in output_file
        assert ".png" in output_file


def test_visualize_transforms_n_2(download_functional_test_files):
    __data_testing_dir__ = Path(__tmp_dir__, "data_functional_testing")
    __input_file__ = Path(__data_testing_dir__, 'sub-unf01/anat/sub-unf01_T1w.nii.gz')
    __output_dir__ = Path(__tmp_dir__, "output_visualize_transforms_n_2")
    __config_file__ = Path(__data_testing_dir__, "model_config.json")
    __label_file__ = Path(__data_testing_dir__,
                                  'derivatives/labels/sub-test001/anat/sub-unf01_T1w_seg-manual.nii.gz')
    visualize_transforms.main(args=['--input', str(__input_file__),
                                    '--output', str(__output_dir__),
                                    '--config', str(__config_file__),
                                    '-r', str(__label_file__),
                                    '-n', '2'])
    assert __output_dir__.exists()
    output_files = [f.name for f in __output_dir__.iterdir()]
    assert len(output_files) == 10
    for output_file in output_files:
        assert "Resample" in output_file
        assert "slice" in output_file
        assert ".png" in output_file


def teardown_function():
    remove_tmp_dir()
