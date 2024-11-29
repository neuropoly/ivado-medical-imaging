import os

from pathlib import Path
from testing.unit_tests.t_utils import path_repo_root


def test_config_json():
    config_path = f"{path_repo_root}/ivadomed/config/"
    for filename in os.listdir(config_path):
        config_file_path = Path(config_path, filename)
        with open(config_file_path, "r") as config_file:
            lines = config_file.readlines()
            for line in lines:
                assert '\t' not in line
