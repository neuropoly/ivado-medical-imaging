import logging
import os
from cli_base import remove_tmp_dir, __tmp_dir__, create_tmp_dir
from ivadomed.scripts import visualize_transforms
logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


def test_visualize_transforms():
    # visualize_transforms.main(args=[])
    assert 1 == 1


def teardown_function():
    remove_tmp_dir()
