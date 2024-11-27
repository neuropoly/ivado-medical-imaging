import logging
import os
import pytest
from pytest_console_scripts import script_runner
from testing.functional_tests.t_utils import __data_testing_dir__, download_functional_test_files, __tmp_dir__, \
    create_tmp_dir, remove_tmp_dir
from testing.common_testing_util import download_dataset, path_repo_root

logger = logging.getLogger(__name__)

def setup_function():
    create_tmp_dir()


@pytest.mark.script_launch_mode('subprocess')
def test_one_class_segmentation_2d_unet(download_functional_test_files, script_runner):
    download_dataset("data_example_spinegeneric")
    file_config = os.path.join(__data_testing_dir__, 'one_class_segmentation_2d_unet_config.json')

    ret = script_runner.run('ivadomed', '--train', '-c', f'{file_config}',
                            '--path-output', f'{__tmp_dir__}',
                            cwd=path_repo_root)

    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success

    ret = script_runner.run('ivadomed', '--test', '-c', f'{file_config}',
                            '--path-output', f'{__tmp_dir__}',
                            cwd=path_repo_root)

    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success


def teardown_function():
    remove_tmp_dir()