import os
from pathlib import Path

import json
import csv_diff
import pandas as pd
import pytest
from loguru import logger

from ivadomed.keywords import BidsDataFrameKW, LoaderParamsKW, ContrastParamsKW, ConfigKW
from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed.scripts import download_data as ivadomed_download_data
from ivadomed.main import set_loader_params
import time
import shutil
import sys

# Cannot use __ivadomed_dir__ here as that would refer to the IvadoMed installed in the Conda/Venv
# Absolute path to the temporary folder, which should sit at the root of the repo.
# We could also replace this hard coded levels reference with a simple loop to search for .git/.github sitting at the
# root of the repo. To be discussed.
FOLDER_LEVELS_FROM_ROOT = 1

path_repo_root: Path = Path(__file__).parents[FOLDER_LEVELS_FROM_ROOT].absolute()

if not sys.platform.startswith('win'):
    path_temp: Path = Path("/tmp")
else:
    path_temp: Path = Path(Path.home().parts[0], 'temp')

timestr = time.strftime("%Y%m%d-%H%M%S")
tmptimestr = "tmp" + timestr
new_path_temp = path_temp / tmptimestr

path_unit_tests: str = str(path_repo_root / "testing" / "unit_tests")
path_data_testing_source: str = str(path_repo_root / "data_testing")
path_data_testing_tmp: str = str(path_temp / tmptimestr / Path(path_data_testing_source).name)

path_functional_tests: str = str(path_repo_root / "testing" / "functional_tests")
path_data_functional_source: str = str(path_repo_root / "data_functional_testing")
path_data_functional_tmp: str = str(path_temp / tmptimestr / Path(path_data_functional_source).name)

path_data_multi_sessions_contrasts_source: Path = path_repo_root / "data_multi_testing"
path_data_multi_sessions_contrasts_tmp: Path = path_temp / tmptimestr / Path(
    path_data_multi_sessions_contrasts_source).name


def download_dataset(dataset: str = 'data_testing'):
    """Download testing data from internet.

    Args:
        dataset (str): the name of the dataset to download
    """

    path_dataset: Path = path_repo_root / dataset

    # Early abort if testing data already exist.
    if path_dataset.exists():
        logger.warning(f"\nTesting data files appear to already exist at {path_dataset}, aborting download")
        return

    logger.info(f"\nDownloading testing data... to {dataset}")

    # Call the ivadomed download CLI
    ivadomed_download_data.main([
        '-d', dataset,
        '-o', str(path_dataset)
    ])


def remove_dataset(dataset: str = 'data_testing'):
    """Recursively remove the data_testing folder.

    Args:
        dataset (str): the name of the dataset to remove
    """

    path_dataset = Path(path_temp, dataset)

    logger.debug(f"rm -rf {path_dataset}")

    shutil.rmtree(path_dataset, ignore_errors=True)


def remove_tmp_dir():
    """Recursively remove the tmp_XXXXX time stamped unit test directory if it exists."""

    tests_count = 0
    for element in list(path_temp.iterdir()):
        if element.is_dir() and element.name.startswith("tmp"):
            tests_count += 1

    # Default Keeping past 21000 unit tests (keep in mind we run about 209 tests per entire pytest run, x 100 runs)
    tests_per_run = 210  # as of 2022-05-30T132815EST we run 209 tests per unit test run.
    runs_to_keep = 100
    max_runs = tests_per_run * runs_to_keep

    if tests_count > max_runs:
        num_to_delete = tests_count - max_runs

        if num_to_delete < 1:
            return

        # Sort lists by runs
        list_test_folder_name = sorted(path_temp.iterdir(), key=os.path.getctime)

        list_delete_folder_name = list_test_folder_name[101:]

        for folder_name in list_delete_folder_name:
            if folder_name.is_dir() and folder_name.stem.startswith("tmp"):
                shutil.rmtree(folder_name, ignore_errors=True)
