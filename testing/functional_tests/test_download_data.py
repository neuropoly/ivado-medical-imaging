import logging
import pytest
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir
from testing.common_testing_util import remove_dataset, remove_tmp_dir
from ivadomed.scripts import download_data
from ivadomed.utils import ArgParseException
from pathlib import Path
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir(copy_data_testing_dir=False)


def test_download_data():
    for dataset in download_data.DICT_URL:
        output_folder = Path(__tmp_dir__, dataset)
        download_data.main(args=['-d', dataset,
                                 '-o', str(output_folder)])
        assert output_folder.exists()
        remove_dataset(dataset=dataset)


def test_download_data_no_dataset_specified():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        download_data.main()


def teardown_function():
    remove_tmp_dir()
