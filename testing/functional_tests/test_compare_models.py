import logging
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, __data_testing_dir__, \
    download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed.scripts import compare_models
from pathlib import Path
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_compare_models(download_functional_test_files):
    __output_file__ = Path(__tmp_dir__, 'comparison_results.csv')
    path_df = Path(__data_testing_dir__, 'temporary_results.csv')
    compare_models.main(args=['-df', str(path_df),
                              '-n', '2',
                              '-o', str(__output_file__)])
    assert __output_file__.exists()


def teardown_function():
    remove_tmp_dir()
