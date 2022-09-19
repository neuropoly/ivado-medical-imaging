import os
import json
import logging

from ivadomed.config_manager import ConfigurationManager
from ivadomed.keywords import ConfigKW, LoaderParamsKW
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, __data_testing_dir__, \
    download_functional_test_files
from testing.common_testing_util import remove_tmp_dir

from ivadomed.scripts import training_curve
from pathlib import Path
from loguru import logger
import pytest
from ivadomed.main import run_command

def setup_function():
    create_tmp_dir()


def test_training_with_filedataset(download_functional_test_files):
    from ivadomed.loader.all_dataset_group import example_AllDatasetGroup_json

    # Build the config file
    path_default_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    with open(path_default_config) as json_file:
        json_data: dict = json.load(json_file)

    # Add the new key to JSON.
    json_data.update(example_AllDatasetGroup_json)

    # Popping out the contract key to enable it to AUTO using the new LoaderConfiguration
    json_data[ConfigKW.LOADER_PARAMETERS].pop(LoaderParamsKW.CONTRAST_PARAMS)

    # Patching in the two required parameters.
    json_data["path_output"] = "pytest_output_folder"
    json_data["log_file"] = "log"

    # Debug print out JSON
    logger.trace(json.dumps(json_data, indent=4))

    # Build loader parameter?

    # Build the model parameters
    # Build the Generalized Loader Configuration

    # Build the example dataset

    # Call ivado cmd_train
    best_training_dice, best_training_loss, best_validation_dice, best_validation_loss = run_command(context=json_data)


@pytest.mark.skip(reason="To be Implemented")
def test_training_with_bidsdataset(download_functional_test_files):
    pass


@pytest.mark.skip(reason="To be Implemented")
def test_training_with_regex_dataset(download_functional_test_files):
    pass


@pytest.mark.skip(reason="To be Implemented")
def test_training_with_consolidated_dataset(download_functional_test_files):
    pass


def teardown_function():
    remove_tmp_dir()
