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
def test_optimize_hyperparameters(download_functional_test_files, script_runner):
    download_dataset("data_example_spinegeneric")
    file_config = os.path.join(__data_testing_dir__, 'optimize_hyperparameters_config.json')
    hyper_config = os.path.join(__data_testing_dir__, 'config_hyper.json')

    ret = script_runner.run('ivadomed_automate_training', '-c', f'{file_config}',
                            '-ch', f'{hyper_config}',
                            '-n', '1',
                            cwd=path_repo_root)

    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success


@pytest.mark.script_launch_mode('subprocess')
def test_optimize_hyperparameters_all_combinations(download_functional_test_files, script_runner):
    download_dataset("data_example_spinegeneric")
    file_config = os.path.join(__data_testing_dir__, 'optimize_hyperparameters_config.json')
    hyper_config = os.path.join(__data_testing_dir__, 'config_hyper.json')

    ret = script_runner.run('ivadomed_automate_training', '-c', f'{file_config}',
                            '-ch', f'{hyper_config}',
                            '-n', '1',
                            '--all-combin',
                            cwd=path_repo_root)

    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success


@pytest.mark.script_launch_mode('subprocess')
def test_optimize_hyperparameters_multiple_parameters(download_functional_test_files, script_runner):
    download_dataset("data_example_spinegeneric")
    file_config = os.path.join(__data_testing_dir__, 'optimize_hyperparameters_config.json')
    hyper_config = os.path.join(__data_testing_dir__, 'config_hyper.json')

    ret = script_runner.run('ivadomed_automate_training', '-c', f'{file_config}',
                            '-ch', f'{hyper_config}',
                            '-n', '1',
                            '--multi-params',
                            cwd=path_repo_root)

    print(f"{ret.stdout}")
    print(f"{ret.stderr}")
    assert ret.success


def teardown_function():
    remove_tmp_dir()