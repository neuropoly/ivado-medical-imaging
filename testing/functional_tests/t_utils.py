import pytest
import shutil
from pathlib import Path
from ivadomed.utils import init_ivadomed
from ivadomed import config_manager as imed_config_manager
from testing.common_testing_util import remove_tmp_dir, path_repo_root, path_temp, path_data_functional_source, \
    path_data_functional_tmp, download_dataset

__test_dir__ = Path(path_repo_root, 'testing/functional_tests')
__data_testing_dir__ = path_data_functional_source
__tmp_dir__ = path_temp

init_ivadomed()


@pytest.fixture(scope='session')
def download_functional_test_files():
    """
    This fixture will attempt to download test data file if there are not present.
    """
    download_dataset("data_functional_testing")


def check_sha256(file_config):
    """
    This function checks if sha256 is generated in according to config file
    """
    initial_config = imed_config_manager.ConfigurationManager(file_config).get_config()
    result = []
    name = "config_file.json"
    for path_object in Path(initial_config["path_output"]).parent.glob("**/*"):
        if path_object.is_file() and name in path_object.name:
            result.append(str(path_object))
    assert result != []
    for generated_config in result:
        config = imed_config_manager.ConfigurationManager(generated_config).get_config()
        assert 'training_sha256' in config


def create_tmp_dir(copy_data_testing_dir=True):
    """Create a temporary directory for data_functional and copy test data files.

    1. Remove the ``tmp`` directory if it exists.
    2. Copy the ``data_functional_testing`` directory to the ``tmp`` directory.

    Any data files created during testing will go into ``tmp`` directory.
    This is created/removed for each test.

    Args:
        copy_data_testing_dir (bool): If true, copy the __data_testing_dir_ref__ folder
            into the ``tmp`` folder.
    """
    remove_tmp_dir()
    Path(path_temp).mkdir(parents=True, exist_ok=True)
    if Path(path_data_functional_source).exists() and copy_data_testing_dir:
        shutil.copytree(path_data_functional_source,
                        path_data_functional_tmp)
