import logging
import pytest
from testing.functional_tests.t_utils import create_tmp_dir, __data_testing_dir__, download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed.scripts import convert_to_onnx
from ivadomed.utils import ArgParseException
from pathlib import Path

logger = logging.getLogger(__name__)

__model_path__ = Path(__data_testing_dir__, 'spinegeneric_model.pt')


def setup_function():
    create_tmp_dir()


def test_convert_to_onnx(download_functional_test_files):
    convert_to_onnx.main(args=['-m', f'{__model_path__}', '-d', '2'])
    assert Path(__data_testing_dir__, 'spinegeneric_model.onnx').exists()


def test_convert_to_onnx_no_model():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        convert_to_onnx.main(args=['-d', '2'])


def test_convert_to_onnx_no_dimension():
    with pytest.raises(ArgParseException, match=r"Error parsing args"):
        convert_to_onnx.main(args=['-m', f'{__model_path__}'])


def teardown_function():
    remove_tmp_dir()
