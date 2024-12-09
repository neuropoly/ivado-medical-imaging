# -*- coding: utf-8 -*-
"""Template for Functional Tests

This file is a template for writing functional tests. It will not be run during testing, but you
can use it as a guide for writing your tests.

GitHub Tests:
    In the file ``ivadomed/.github/workflows/run_tests.yml``, you will find how the tests are
    run when you submit a pull request.
    1. ``ivadomed_download_data -d data_functional_testing -o data_functional_testing``
    2. ``pytest --cov=./ --cov-report term-missing``

Running Locally:
    To run tests locally, you will need to download the testing folder and then run the tests:
    1. ``ivadomed_download_data -d data_functional_testing -o data_functional_testing``
    2. ``pytest``

    You can also run just one folder, or just one file:
    ``pytest testing/functional_tests/``
    ``pytest testing/functional_tests/test_example.py``

Folder Structure:
    After downloading the ``data_functional_testing`` folder, your directory should look like this:
    ::

        ivadomed/
        | --- data_functional_testing/
        | --- testing/
        |     | --- unit_tests/
        |     | --- functional_tests/

    When you call ``create_tmp_dir`` in the ``setup_function``, you will copy
    ``data_functional_testing`` to a new ``tmp`` folder:
    ::

        ivadomed/
        | --- data_functional_testing/
        | --- testing/
        |     | --- unit_tests/
        |     | --- functional_tests/
        | --- tmp/
        |     | --- data_functional_testing/

    At the end of each test, the ``tmp`` folder is removed by ``remove_tmp_dir`` in the
    ``teardown_function``. Note that only ``tmp/data_functional_testing`` is removed, not
    ``data_functional_testing``; this is so that you don't need to keep downloading it each time.

    ``__data_testing_dir__``: ``tmp/data_functional_testing``
    ``__tmp_dir__``: ``tmp``

Names and Files:
    Generally speaking, one test file should correspond to one file in the ``ivadomed`` package.
    Tests files should be named: ``test_my_file.py``.

"""

import logging
from testing.functional_tests.t_utils import create_tmp_dir, __data_testing_dir__, __tmp_dir__
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path
logger = logging.getLogger(__name__)


def setup_function():
    """Function which is run before each test in this file.

    ``create_tmp_dir`` will do the following:
    1. Create a directory called ``tmp`` (overwrite if already exists)
    2. Copy ``data_functional_testing`` -> ``tmp/data_functional_testing``

    Add any other things for setup here.
    """
    create_tmp_dir()


def test_template():
    # Test Input Files: all test input files should be in tmp/data_functional_testing
    # aka __data_testing_dir__
    logger.info([f.name for f in Path(__data_testing_dir__).iterdir()])

    # Test Output Files: put your output files in tmp folder
    Path(__tmp_dir__, 'my_output_dir').mkdir(parents=True, exist_ok=True)

    assert 1 == 1


def teardown_function():
    """Function which is run after each test in this file.

    ``remove_tmp_dir`` will do the following:
    1. Delete the directory called ``tmp`` (if already exists)

    Add any other things for teardown here.
    Note: this function gets run after each test, so files/data will not be saved
        in between tests.
    """
    remove_tmp_dir()
