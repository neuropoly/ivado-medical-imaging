"""Unit Test for Automate Training

In ``automate_training.py``, we first create a ``param_list``, and then create a ``config_list``.
For testing, we are using the initial ``config``:

.. code-block:: JSON

    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "DiceLoss"},
            "scheduler": {
                "initial_lr": 0.001
            }
        },
        "default_model":     {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/",
        "gpu_ids": [1]
    }

and the ``config_hyper``:

.. code-block:: JSON

    {
        "training_parameters": {
            "batch_size": [2, 64],
            "loss": [
                {"name": "DiceLoss"},
                {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5}
             ]
        },
        "default_model": {
            "depth": [2, 3, 4]
        },
        "gpu_ids": [[2], [5]]
    }

The ``config_list`` depends on the flag ``all-combin``, ``multi-params``, or no flag.

For no flag (``default``), the options are:

.. code-block::

    batch_size = 2, loss = "DiceLoss", depth = 3, gpu_ids = [1]
    batch_size = 64, loss = "DiceLoss", depth = 3, gpu_ids = [1]
    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [1]
    batch_size = 18, loss = "FocalLoss", depth = 3, gpu_ids = [1]
    batch_size = 18, loss = "DiceLoss", depth = 2, gpu_ids = [1]
    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [1]
    batch_size = 18, loss = "DiceLoss", depth = 4, gpu_ids = [1]
    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [2]
    batch_size = 18, loss = "DiceLoss", depth = 3, gpu_ids = [5]

For ``all_combin``:

.. code-block::

    batch_size = 2, loss = "DiceLoss", depth = 2, gpu_ids = [2]
    batch_size = 64, loss = "DiceLoss", depth = 2, gpu_ids = [2]
    batch_size = 2, loss = "FocalLoss", depth = 2, gpu_ids = [2]
    batch_size = 64, loss = "FocalLoss", depth = 2, gpu_ids = [2]
    batch_size = 2, loss = "DiceLoss", depth = 3, gpu_ids = [2]
    batch_size = 64, loss = "DiceLoss", depth = 3, gpu_ids = [2]
    batch_size = 2, loss = "FocalLoss", depth = 3, gpu_ids = [2]
    batch_size = 64, loss = "FocalLoss", depth = 3, gpu_ids = [2]
    batch_size = 2, loss = "DiceLoss", depth = 2, gpu_ids = [5]
    batch_size = 64, loss = "DiceLoss", depth = 2, gpu_ids = [5]
    batch_size = 2, loss = "FocalLoss", depth = 2, gpu_ids = [5]
    batch_size = 64, loss = "FocalLoss", depth = 2, gpu_ids = [5]
    batch_size = 2, loss = "DiceLoss", depth = 3, gpu_ids = [5]
    batch_size = 64, loss = "DiceLoss", depth = 3, gpu_ids = [5]
    batch_size = 2, loss = "FocalLoss", depth = 3, gpu_ids = [5]
    batch_size = 64, loss = "FocalLoss", depth = 3, gpu_ids = [5]

For ``mult_params``:

.. code-block::

    batch_size = 2, loss = "DiceLoss", depth = 2, gpu_ids = [2]
    batch_size = 64, loss = "FocalLoss", depth = 3, gpu_ids = [5]

"""

import pytest

from ivadomed.loader.bids_dataframe import BidsDataframe
from ivadomed.scripts.automate_training import make_config_list, get_param_list, \
    HyperparameterOption
from ivadomed.utils import generate_sha_256
from loguru import logger
from testing.unit_tests.t_utils import create_tmp_dir, __data_testing_dir__, __tmp_dir__, download_data_testing_test_files
from testing.common_testing_util import remove_tmp_dir

initial_config = {
    "training_parameters": {
        "batch_size": 18,
        "loss": {"name": "DiceLoss"},
        "scheduler": {
            "initial_lr": 0.001
        }
    },
    "default_model":     {
        "name": "Unet",
        "dropout_rate": 0.3,
        "depth": 3
    },
    "path_output": "./tmp/",
    "gpu_ids": [1]
}

expected_config_list_all_combin = [
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-2-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-3-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-4-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-2-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-3-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-4-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [2],
        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-2"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-2-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-3-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-4-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-2-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-2-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-3-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-64-loss-name-DiceLoss-depth-4-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-2-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-5"
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "gpu_ids": [5],
        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-4-gpu_ids-5"
    }
]


expected_config_list_neither = [
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-batch_size-2",
        "gpu_ids": [1]
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-batch_size-64",
        "gpu_ids": [1]
    },

    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-loss-name-DiceLoss",
        "gpu_ids": [1]
    },
    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-loss-name-FocalLoss-gamma-0.2-alpha-0.5",
        "gpu_ids": [1]
     },
    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "path_output": "./tmp/-depth-2",
        "gpu_ids": [1]
    },
    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-depth-3",
        "gpu_ids": [1]
    },
    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 4
        },
        "path_output": "./tmp/-depth-4",
        "gpu_ids": [1]
    },
    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-gpu_ids-2",
        "gpu_ids": [2]
    },
    {
        "training_parameters": {
            "batch_size": 18,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-gpu_ids-5",
        "gpu_ids": [5]
    }
]

expected_config_list_multi_params = [
    {
        "training_parameters": {
            "batch_size": 2,
            "loss": {"name": "DiceLoss"},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 2
        },
        "path_output": "./tmp/-batch_size-2-loss-name-DiceLoss-depth-2-gpu_ids-2",
        "gpu_ids": [2]
    },
    {
        "training_parameters": {
            "batch_size": 64,
            "loss": {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5},
            "scheduler": {"initial_lr": 0.001}
        },
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "depth": 3
        },
        "path_output": "./tmp/-batch_size-64-loss-name-FocalLoss-gamma-0.2-alpha-0.5-depth-3-gpu_ids-5",
        "gpu_ids": [5]
    }
]

expected_param_list = [
    HyperparameterOption("batch_size", {"training_parameters": {"batch_size": 2}}, 2),
    HyperparameterOption("batch_size", {"training_parameters": {"batch_size": 64}}, 64),
    HyperparameterOption("loss", {"training_parameters": {"loss": {"name": "DiceLoss"}}},
                         {"name": "DiceLoss"}),
    HyperparameterOption("loss", {"training_parameters": {"loss": {"name": "FocalLoss",
                                  "gamma": 0.2, "alpha": 0.5}}},
                         {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5}),
    HyperparameterOption("depth", {"default_model": {"depth": 2}}, 2),
    HyperparameterOption("depth", {"default_model": {"depth": 3}}, 3),
    HyperparameterOption("depth", {"default_model": {"depth": 4}}, 4),
    HyperparameterOption("gpu_ids", {"gpu_ids": [2]}, [2]),
    HyperparameterOption("gpu_ids", {"gpu_ids": [5]}, [5])
]


def setup_function():
    create_tmp_dir()


@pytest.mark.parametrize("config_hyper", [
    {
        "training_parameters": {
            "batch_size": [2, 64],
            "loss": [
                {"name": "DiceLoss"},
                {"name": "FocalLoss", "gamma": 0.2, "alpha": 0.5}
             ]
        },
        "default_model": {
            "depth": [2, 3, 4]
        },
        "gpu_ids": [[2], [5]]
    }
])
@pytest.mark.parametrize(
    "expected_param_list",
    [
        pytest.param(
            expected_param_list
        )
    ]
)
def test_get_param_list(config_hyper, expected_param_list):
    param_list = get_param_list(config_hyper, [], [])
    assert param_list == expected_param_list


@pytest.mark.parametrize("initial_config", [initial_config])
@pytest.mark.parametrize(
    "all_combin,multi_params,param_list,expected_config_list",
    [
        pytest.param(
            False, False,
            expected_param_list,
            expected_config_list_neither,
            id="neither"
        ),
        pytest.param(
            True, False,
            expected_param_list,
            expected_config_list_all_combin,
            id="all_combin"
        ),
        pytest.param(
            False, True,
            expected_param_list,
            expected_config_list_multi_params,
            id="multi_params"
        )
    ]
)
def test_make_config_list(initial_config, all_combin, multi_params, param_list,
                          expected_config_list):
    config_list = make_config_list(param_list, initial_config, all_combin, multi_params)
    assert len(config_list) == len(expected_config_list)
    for config_option in config_list:
        if config_option not in expected_config_list:
            logger.debug(config_option['path_output'])
            logger.debug(expected_config_list)
        assert config_option in expected_config_list
    for config_option in expected_config_list:
        assert config_option in config_list


@pytest.mark.parametrize("initial_config", [initial_config])
def test_config_sha256(download_data_testing_test_files, initial_config):
    file_lst = ["sub-unf01_T2w.nii.gz"]
    loader_params = {
        "transforms_params": {},
        "data_list": ['sub-unf01'],
        "dataset_type": "testing",
        "requires_undo": True,
        "contrast_params": {"contrast_lst": ['T2w'], "balance": {}},
        "path_data": [__data_testing_dir__],
        "target_suffix": ["_lesion-manual"],
        "extensions": [".nii.gz"],
        "roi_params": {"suffix": "_seg-manual", "slice_filter_roi": 10},
        "slice_filter_params": {
            "filter_empty_mask": False,
            "filter_empty_input": True
        },
        "slice_axis": "axial",
        "multichannel": False
    }

    bids_df = BidsDataframe(loader_params, __tmp_dir__, derivatives=True)
    generate_sha_256(initial_config, bids_df.df, file_lst)
    assert(initial_config['training_sha256']['sub-unf01_T2w.nii.gz'] ==
           'f020b368fea15399fa112badd28b2df69e044dba5d23b3fe1646d12d7d3d39ac')


def teardown_function():
    remove_tmp_dir()
