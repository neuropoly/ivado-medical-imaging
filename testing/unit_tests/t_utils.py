import os
import shutil
import pytest
from pathlib import Path
from ivadomed.utils import init_ivadomed
from testing.common_testing_util import remove_tmp_dir, path_repo_root, path_temp, path_data_testing_tmp, \
    path_data_testing_source, download_dataset, path_data_multi_sessions_contrasts_source, \
    path_data_multi_sessions_contrasts_tmp

__test_dir__ = Path(path_repo_root, 'testing/unit_tests')
__data_testing_dir__ = path_data_testing_tmp
__tmp_dir__ = path_temp

init_ivadomed()


@pytest.fixture(scope='session')
def download_data_testing_test_files():
    """
    This fixture will attempt to download test data file if there are not present.
    """
    download_dataset("data_testing")


@pytest.fixture(scope='session')
def download_multi_data():
    """
    This Pytest fixture DOWNLOAD all the test data set REQUIRED for the multi-session, multi-contrast related unit
    testing.
    """
    download_dataset("data_multi_testing")


def create_tmp_dir(copy_data_testing_dir=True):
    """Create a temporary directory for unit_test data and copy test data files.

    1. Remove the ``tmp`` directory if it exists.
    2. Copy the ``data_testing`` directory to the ``tmp`` directory.

    Any data files created during testing will go into ``tmp`` directory.
    This is created/removed for each test.

    Args:
        copy_data_testing_dir (bool): If true, copy the __data_testing_dir_ref__ folder
            into the ``tmp`` folder.
    """
    remove_tmp_dir()
    Path(path_temp).mkdir()
    if Path(path_data_testing_source).exists() and copy_data_testing_dir:
        shutil.copytree(path_data_testing_source,
                        path_data_testing_tmp)


def create_tmp_dir_multi_session():
    """Create a temporary directory for data related to multi-session unit tests and copy test data files.
    1. Remove the ``tmp`` directory if it exists.
    2. Copy the ``data_testing_multi`` directory to the ``tmp`` directory. Ignoring the `.git` folder within
    Any data files created during testing will go into ``tmp`` directory.
    This is created/removed for each test.
    """
    ignore_git_pattern = shutil.ignore_patterns(str(path_data_multi_sessions_contrasts_source / '.git'))
    remove_tmp_dir()
    Path(path_temp).mkdir()
    if Path(path_data_multi_sessions_contrasts_source).exists():
        shutil.copytree(path_data_multi_sessions_contrasts_source,
                        path_data_multi_sessions_contrasts_tmp,
                        ignore=ignore_git_pattern)