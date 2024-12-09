import json
import logging
import os
import pytest
from pytest_console_scripts import script_runner
from pathlib import Path
from testing.functional_tests.t_utils import __tmp_dir__, create_tmp_dir, __data_testing_dir__, \
    download_functional_test_files
from testing.common_testing_util import remove_tmp_dir
from ivadomed import config_manager as imed_config_manager
from ivadomed.keywords import ConfigKW, ModelParamsKW, LoaderParamsKW, ContrastParamsKW, TransformationKW


logger = logging.getLogger(__name__)


def setup_function():
    create_tmp_dir()


@pytest.mark.script_launch_mode('subprocess')
def test_training_3d_1class_single_channel_with_data_augmentation(download_functional_test_files, script_runner):

    # Load automate training config as context
    file_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    context = imed_config_manager.ConfigurationManager(file_config).get_config()

    # Modify key-value pairs in context for given test
    # Set-up 3D model params
    context[ConfigKW.DEFAULT_MODEL][ModelParamsKW.IS_2D] = False
    context[ConfigKW.MODIFIED_3D_UNET] = {
        ModelParamsKW.APPLIED: True,
        ModelParamsKW.LENGTH_3D: [32, 32, 16],
        ModelParamsKW.STRIDE_3D: [32, 32, 16],
        ModelParamsKW.N_FILTERS: 4
    }
    # Set target_suffix (1 class)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.TARGET_SUFFIX] = ["_lesion-manual"]
    # Set contrasts or interest (2 single channels)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TRAINING_VALIDATION] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TESTING] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.MULTICHANNEL] = False
    # Set 3D preprocessing and data augmentation
    context[ConfigKW.TRANSFORMATION][TransformationKW.RESAMPLE] = {
        "wspace": 0.75,
        "hspace": 0.75,
        "dspace": 0.75
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.CENTERCROP] = {
        "size": [32, 32, 16],
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.RANDOM_AFFINE] = {
        "degrees": 10,
        "scale": [0.03, 0.03, 0.03],
        "translate": [0.8, 0.8, 0.8],
        "applied_to": ["im", "gt"],
        "dataset_type": ["training"]
    }

    # Write temporary config file for given test
    file_config_updated = os.path.join(__tmp_dir__, "data_functional_testing", "config_3d_training.json")
    with Path(file_config_updated).open(mode='w') as fp:
        json.dump(context, fp, indent=4)

    # Set output directory
    __output_dir__ = Path(__tmp_dir__, 'results')

    # Run ivadomed
    ret = script_runner.run('ivadomed', '-c', f'{file_config_updated}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--path-output', f'{__output_dir__}')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success


@pytest.mark.script_launch_mode('subprocess')
def test_training_3d_2class_single_channel_with_data_augmentation(download_functional_test_files, script_runner):

    # Load automate training config as context
    file_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    context = imed_config_manager.ConfigurationManager(file_config).get_config()

    # Modify key-value pairs in context for given test
    # Set-up 3D model params
    context[ConfigKW.DEFAULT_MODEL][ModelParamsKW.IS_2D] = False
    context[ConfigKW.MODIFIED_3D_UNET] = {
        ModelParamsKW.APPLIED: True,
        ModelParamsKW.LENGTH_3D: [32, 32, 16],
        ModelParamsKW.STRIDE_3D: [32, 32, 16],
        ModelParamsKW.N_FILTERS: 4
    }
    # Set target_suffix (2-class)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.TARGET_SUFFIX] = ["_lesion-manual", "_seg-manual"]
    # Set contrasts or interest (2 single channels)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TRAINING_VALIDATION] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TESTING] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.MULTICHANNEL] = False
    # Set 3D preprocessing and data augmentation
    context[ConfigKW.TRANSFORMATION][TransformationKW.RESAMPLE] = {
        "wspace": 0.75,
        "hspace": 0.75,
        "dspace": 0.75
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.CENTERCROP] = {
        "size": [32, 32, 16],
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.RANDOM_AFFINE] = {
        "degrees": 10,
        "scale": [0.03, 0.03, 0.03],
        "translate": [0.8, 0.8, 0.8],
        "applied_to": ["im", "gt"],
        "dataset_type": ["training"]
    }

    # Write temporary config file for given test
    file_config_updated = os.path.join(__tmp_dir__, "data_functional_testing", "config_3d_training.json")
    with Path(file_config_updated).open(mode='w') as fp:
        json.dump(context, fp, indent=4)

    # Set output directory
    __output_dir__ = Path(__tmp_dir__, 'results')

    # Run ivadomed
    ret = script_runner.run('ivadomed', '-c', f'{file_config_updated}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--path-output', f'{__output_dir__}')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success


@pytest.mark.script_launch_mode('subprocess')
def test_training_3d_1class_multi_channel_with_data_augmentation(download_functional_test_files, script_runner):

    # Load automate training config as context
    file_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    context = imed_config_manager.ConfigurationManager(file_config).get_config()

    # Modify key-value pairs in context for given test
    # Set-up 3D model params
    context[ConfigKW.DEFAULT_MODEL][ModelParamsKW.IS_2D] = False
    context[ConfigKW.MODIFIED_3D_UNET] = {
        ModelParamsKW.APPLIED: True,
        ModelParamsKW.LENGTH_3D: [32, 32, 16],
        ModelParamsKW.STRIDE_3D: [32, 32, 16],
        ModelParamsKW.N_FILTERS: 4
    }
    # Set target_suffix (1-class)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.TARGET_SUFFIX] = ["_lesion-manual"]
    # Set contrasts or interest (1 multi-channel)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TRAINING_VALIDATION] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TESTING] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.MULTICHANNEL] = True
    # Set 3D preprocessing and data augmentation
    context[ConfigKW.TRANSFORMATION][TransformationKW.RESAMPLE] = {
        "wspace": 0.75,
        "hspace": 0.75,
        "dspace": 0.75
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.CENTERCROP] = {
        "size": [32, 32, 16],
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.RANDOM_AFFINE] = {
        "degrees": 10,
        "scale": [0.03, 0.03, 0.03],
        "translate": [0.8, 0.8, 0.8],
        "applied_to": ["im", "gt"],
        "dataset_type": ["training"]
    }

    # Write temporary config file for given test
    file_config_updated = os.path.join(__tmp_dir__, "data_functional_testing", "config_3d_training.json")
    with Path(file_config_updated).open(mode='w') as fp:
        json.dump(context, fp, indent=4)

    # Set output directory
    __output_dir__ = Path(__tmp_dir__, 'results')

    # Run ivadomed
    ret = script_runner.run('ivadomed', '-c', f'{file_config_updated}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--path-output', f'{__output_dir__}')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success


@pytest.mark.script_launch_mode('subprocess')
def test_training_3d_1class_multirater_with_data_augmentation(download_functional_test_files, script_runner):

    # Load automate training config as context
    file_config = os.path.join(__data_testing_dir__, 'automate_training_config.json')
    context = imed_config_manager.ConfigurationManager(file_config).get_config()

    # Modify key-value pairs in context for given test
    # Set-up 3D model params
    context[ConfigKW.DEFAULT_MODEL][ModelParamsKW.IS_2D] = False
    context[ConfigKW.MODIFIED_3D_UNET] = {
        ModelParamsKW.APPLIED: True,
        ModelParamsKW.LENGTH_3D: [32, 32, 16],
        ModelParamsKW.STRIDE_3D: [32, 32, 16],
        ModelParamsKW.N_FILTERS: 4
    }
    # Set target_suffix (1-class, multirater)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.TARGET_SUFFIX] = [["_lesion-manual", "_seg-manual"]]
    # Set contrasts or interest (2 single channels)
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TRAINING_VALIDATION] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.CONTRAST_PARAMS][ContrastParamsKW.TESTING] = ["T1w", "T2w"]
    context[ConfigKW.LOADER_PARAMETERS][LoaderParamsKW.MULTICHANNEL] = False
    # Set 3D preprocessing and data augmentation
    context[ConfigKW.TRANSFORMATION][TransformationKW.RESAMPLE] = {
        "wspace": 0.75,
        "hspace": 0.75,
        "dspace": 0.75
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.CENTERCROP] = {
        "size": [32, 32, 16],
    }
    context[ConfigKW.TRANSFORMATION][TransformationKW.RANDOM_AFFINE] = {
        "degrees": 10,
        "scale": [0.03, 0.03, 0.03],
        "translate": [0.8, 0.8, 0.8],
        "applied_to": ["im", "gt"],
        "dataset_type": ["training"]
    }

    # Write temporary config file for given test
    file_config_updated = os.path.join(__tmp_dir__, "data_functional_testing", "config_3d_training.json")
    with Path(file_config_updated).open(mode='w') as fp:
        json.dump(context, fp, indent=4)

    # Set output directory
    __output_dir__ = Path(__tmp_dir__, 'results')

    # Run ivadomed
    ret = script_runner.run('ivadomed', '-c', f'{file_config_updated}',
                            '--path-data', f'{__data_testing_dir__}',
                            '--path-output', f'{__output_dir__}')
    logger.debug(f"{ret.stdout}")
    logger.debug(f"{ret.stderr}")
    assert ret.success


def teardown_function():
    remove_tmp_dir()
