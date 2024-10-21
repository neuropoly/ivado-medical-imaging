Configuration File
==================

All parameters used for loading data, training and predicting are contained
within a single JSON configuration file. This section describes how to set up
this configuration file.

For convenience, here is an generic configuration file:
`config\_config.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config.json>`__.

Below are other, more specific configuration files:

- `config\_classification.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_classification.json>`__: Trains a classification model.

- `config\_sctTesting.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_sctTesting.json>`__: Trains a 2D segmentation task with the U-Net architecture.

- `config\_spineGeHemis.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_spineGeHemis.json>`__: Trains a segmentation task with the HeMIS-UNet architecture.

- `config\_tumorSeg.json <https://raw.githubusercontent.com/ivadomed/ivadomed/master/ivadomed/config/config_tumorSeg.json>`__: Trains a segmentation task with a 3D U-Net architecture.


General Parameters
------------------


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "command",
        "description": "Run the specified command.",
        "type": "string",
        "options": {"train": "train a model on a training/validation sub-datasets",
                    "test": "evaluate a trained model on a testing sub-dataset",
                    "segment": "segment a entire dataset using a trained model"
        }
    }

.. code-block:: JSON

    {
        "command": "train"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "gpu_ids",
        "description": "List of IDs of one or more GPUs to use. Default: ``[0]``.",
        "type": "list[int]"
    }

.. code-block:: JSON

    {
        "gpu_ids": [1,2,3]
    }

.. note::
    Currently only ``ivadomed_automate_training`` supports the use of more than one GPU.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "path_output",
        "description": "Folder name that will contain the output files (e.g., trained model,
            predictions, results).",
        "type": "string"
    }




.. code-block:: JSON

    {
        "path_output": "tmp/spineGeneric"
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "model_name",
        "description": "Folder name containing the trained model (ONNX format) and its configuration
            file, located within ``log_directory/``.",
        "type": "string"
    }



.. code-block:: sh

    "log_directory/seg_gm_t2star/seg_gm_t2star.onnx"
    "log_directory/seg_gm_t2star/seg_gm_t2star.json"

When possible, the folder name will follow the following convention:
``task_(animal)_region_(contrast)`` with

.. code-block:: sh

   task = {seg, label, find}
   animal = {human, dog, cat, rat, mouse, ...}
   region = {sc, gm, csf, brainstem, ...}
   contrast = {t1, t2, t2star, dwi, ...}


.. code-block:: JSON

   {
       "model_name": "seg_gm_t2star"
   }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "debugging",
        "description": "Extended verbosity and intermediate outputs. Default: ``False``.",
        "type": "boolean"
    }



.. code-block:: JSON

    {
        "debugging": true
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "log_file",
        "description": "Name of the file to be logged to, located within ``log_directory/``. Default: ``log``.",
        "type": "string"
    }



.. code-block:: JSON

    {
        "log_file": "log"
    }


Weights & Biases (WandB)
------------------------

WandB is an additional option to track your DL experiments. It provides a
feature-rich dashboard (accessible through any web-browser) to track and visualize the learning
curves, gradients, and media. It is recommended to setup a personal
WandB account to track experiments on WandB, however, you can still train ivadomed models
without an account, since the metrics are logged on Tensorboard by default.



.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "wandb_api_key",
        "$$description": [
            "A private key used to sync the local wandb folder with the wandb dashboard accessible through the browser.\n",
            "The API key can be found from the browser in your WandB Account's Settings, under the section ``API Keys``.\n",
            "Note that once it is successfully authenticated, a message would be printed in the terminal notifying\n",
            "that the API key is stored in the ``.netrc`` file in the ``/home`` folder.
        ],
        "type": "string"
    }

.. code-block:: JSON

    {
        "wandb": {
            "wandb_api_key": "<alphanumeric-key-here>"
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "project_name",
        "$$description": [
            "Defines the name of the current project to which the groups and runs will be synced. Default: ``my_project``."
        ],
        "type": "string"
    }

.. code-block:: JSON

    {
        "wandb": {
            "project_name": "my-temp-project"
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "group_name",
        "$$description": [
            "Defines the name of the group to which the runs will be synced. On the WandB Dashboard,\n",
            "the groups can be found on clicking the ``Projects`` tab on the left. Default: ``my_group``."
        ],
        "type": "string"
    }

.. code-block:: JSON

    {
        "wandb": {
            "group_name": "my-temp-group"
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "run_name",
        "$$description": [
            "Defines the name of the current run (or, experiment). All the previous and active runs\n",
            "can be found under the corresponding group on the WandB Dashboard. Default: ``run-1``."
        ],
        "type": "string"
    }

.. code-block:: JSON

    {
        "wandb": {
            "run_name": "run-1"
        }
    }

.. note::
    Using the same ``run_name`` does not replace the previous run but does create multiple entries of the same name. If left empty then the default is a random string assigned by WandB.

.. note::
    We recommend defining the project/group/run names such that hierarchy is easily understandable. For instance, ``project_name`` could be the name of the dataset or the problem you are working (i.e. brain tumor segmentation/spinal cord lesion segmentation etc.), the ``group_name`` could be the various models you are willing to train, and the ``run_name`` could be the various experiments within a particular model (i.e. typically with different hyperparameters).

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "log_grads_every",
        "$$description": [
            "Defines the frequency (in number of steps) of the logging of gradients on to the Dashboard to track and visualize\n",
            "their histograms as the model trains. Default: ``100``.\n"
        ],
        "type": "int"
    }

.. code-block:: JSON

    {
        "wandb": {
            "log_grads_every": 100
        }
    }

.. note::
    There are two important points to be noted:
    (1) Gradients can be large so they can consume more storage space if ``log_grads_every`` is set to a small number,
    (2) ``log_grads_every`` also depends on the total duration of training, i.e. if the model is run for only
    a few epochs, gradients might not be logged if ``log_grads_every`` is too large. Hence, the right frequency of
    gradient logging depends on the training duration and model size.

.. note::
    If ``debugging = True`` is specified in the config file, the training and validation input images, ground truth labels, and
    the model predictions are also periodically logged on WandB, which can be seen under ``Media`` on the WandB Dashboard.




Loader Parameters
-----------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "path_data",
        "description": "Path(s) of the BIDS folder(s).",
        "type": "str or list[str]"
    }


.. code-block:: JSON

    {
        "loader_parameters": {
            "path_data": ["path/to/data_example_spinegeneric", "path/to/other_data_example"]
        }
    }

Alternatively:


.. code-block:: JSON

    {
        "loader_parameters": {
            "path_data": "path/to/data_example_spinegeneric"
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "bids_config",
        "$$description": [
            "(Optional). Path of the custom BIDS configuration file for",
            "BIDS non-compliant modalities. Default: ``ivadomed/config/config_bids.json``."
        ],
        "type": "string"
    }



.. code-block:: JSON

    {
        "loader_parameters": {
            "bids_config": "ivadomed/config/config_bids.json"
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "subject_selection",
        "description": "Used to specify a custom subject selection from a dataset.",
        "type": "dict",
        "options": {
            "n": {
                "description": "List containing the number subjects of each metadata. Default: ``[]``.",
                "type": "list[int]"
            },
            "metadata": {
                "$$description": [
                    "List of metadata used to select the subjects. Each metadata should be the name\n",
                    "of a column from the participants.tsv file. Default: ``[]``."
                ],
                "type": "list[str]"
            },
            "value": {
                "description": "List of metadata values of the subject to be selected. Default: ``[]``.",
                "type": "list[str]"
            }
        }
    }




.. code-block:: JSON

    {
        "loader_parameters": {
            "subject_selection": {
                "n": [5, 10],
                "metadata": ["disease", "disease"],
                "value": ["healthy", "ms"]
            }
        }
    }

In this example, a subdataset composed of 5 healthy subjects and 10 ms subjects will be selected
for training/testing.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "target_suffix",
        "description": "Suffix list of the derivative file containing the ground-truth of interest. Default: ``[]``.",
        "type": "list[str]"
    }



.. code-block:: JSON

    {
        "loader_parameters": {
            "target_suffix": ["_seg-manual", "_lesion-manual"]
        }
    }

The length of this list controls the number of output channels of the model (i.e.
``out_channel``). If the list has a length greater than 1, then a
multi-class model will be trained. If a list of list(s) is input for a
training, (e.g. [[``"_seg-manual-rater1"``, ``"_seg-manual-rater2"``],
[``"_lesion-manual-rater1"``, ``"_lesion-manual-rater2"``]), then each
sublist is associated with one class but contains the annotations from
different experts: at each training iteration, one of these annotations
will be randomly chosen.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "extensions",
        "$$description": [
            "Used to specify a list of file extensions to be selected for training/testing.\n",
            "Must include the file extensions of both the raw data and derivatives. Default: ``[]``."
        ],
        "type": "list[str]"
    }



.. code-block:: JSON

    {
        "loader_parameters": {
            "extensions": [".png"]
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "contrast_params",
        "type": "dict",
        "options": {
            "training_validation": {
                "type": "list[str]",
                "$$description": [
                    "List of image contrasts (e.g. ``T1w``, ``T2w``) loaded for the training and\n",
                    "validation. If ``multichannel`` is ``true``, this list represents the different\n",
                    "channels of the input tensors (i.e. its length equals model's ``in_channel``).\n",
                    "Otherwise, the contrasts are mixed and the model has only one input channel\n",
                    "(i.e. model's ``in_channel=1``)."
                ]
            },
            "testing": {
                "type": "list[str]",
                "$$description": [
                    "List of image contrasts (e.g. ``T1w``, ``T2w``) loaded in the testing dataset.\n",
                    "Same comment as for ``training_validation`` regarding ``multichannel``."
                ]
            },
            "balance": {
                "type": "dict",
                "$$description": [
                    "Enables to weight the importance of specific channels (or contrasts) in the\n",
                    "dataset: e.g. ``{'T1w': 0.1}`` means that only 10% of the available ``T1w``\n",
                    "images will be included into the training/validation/test set. Please set\n",
                    "``multichannel`` to ``false`` if you are using this parameter. Default: ``{}``."
                ]
            }
        }
    }



.. code-block:: JSON

    {
        "loader_parameters": {
            "contrast_params": {
                "training_validation": ["T1w", "T2w", "T2star"],
                "testing": ["T1w", "T2w", "T2star"],
                "balance": {}
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "multichannel",
        "description": "Indicated if more than a contrast (e.g. ``T1w`` and ``T2w``) is
            used by the model. Default: ``False``.",
        "type": "boolean"
    }

See details in both ``training_validation`` and ``testing`` for the contrasts that are input.



.. code-block:: JSON

    {
        "loader_parameters": {
            "multichannel": false
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "bids_validate",
        "description": "Indicates if the loader should validate the dataset for compliance with BIDS. Default: ``True``.",
        "type": "boolean"
    }



.. code-block:: JSON

    {
        "loader_parameters": {
            "bids_validate": true
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "slice_axis",
        "description": "Sets the slice orientation for 3D NIfTI files on which the model will be used. Default: ``axial``.",
        "type": "string",
        "options": {"sagittal": "plane dividing body into left/right",
                    "coronal": "plane dividing body into front/back",
                    "axial": "plane dividing body into top/bottom"
        }
    }



.. code-block:: JSON

    {
        "loader_parameters": {
            "slice_axis": "sagittal"
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "slice_filter_params",
        "$$description": [
            "Discard a slice from the dataset if it meets a condition, defined below.\n",
            "A slice is an entire 2D image taken from a 3D volume (e.g. an image of size 128x128 taken from a volume of size 128x128x16).\n",
            "Therefore, the parameter ``slice_filter_params`` is applicable for 2D models only.",
        ],
        "type": "dict",
        "options": {
            "filter_empty_input": {
                "type": "boolean",
                "description": "Discard slices where all voxel intensities are zeros. Default: ``True``."
            },
            "filter_empty_mask": {
                "type": "boolean",
                "description": "Discard slices where all voxel labels are zeros. Default: ``False``."
            },
            "filter_absent_class": {
                "type": "boolean",
                "$$description": [
                    "Discard slices where all voxel labels are zero for one or more classes\n",
                    "(this is most relevant for multi-class models that need GT for all classes at training time). Default: ``False``."
                ]
            },
            "filter_classification": {
                "type": "boolean",
                "$$description": [
                    "Discard slices where all images fail a custom classifier filter. If used,\n",
                    "``classifier_path`` must also be specified, pointing to a saved PyTorch classifier. Default: ``False``."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "loader_parameters": {
            "slice_filter_params": {
                "filter_empty_mask": false,
                "filter_empty_input": true
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "patch_filter_params",
        "$$description": [
            "Discard a 2D or 3D patch from the dataset if it meets a condition at training time, defined below.\n",
            "A 2D patch is a portion of a 2D image (e.g. a patch of size 32x32 taken inside an image of size 128x128).\n",
            "A 3D patch is a portion of a 3D volume (e.g. a patch of size 32x32x16 from a volume of size 128x128x16).\n",
            "Therefore, the parameter ``patch_filter_params`` is applicable for 2D or 3D models.\n",
            "In addition, contrary to ``slice_filter_params`` which applies at training and testing time, ``patch_filter_params``\n",
            "is applied only at training time. This is because the reconstruction algorithm for predictions from patches\n",
            "needs to have the predictions for all patches at testing time."
        ],
        "type": "dict",
        "options": {
            "filter_empty_input": {
                "type": "boolean",
                "description": "Discard 2D or 3D patches where all voxel intensities are zeros. Default: ``False``."
            },
            "filter_empty_mask": {
                "type": "boolean",
                "description": "Discard 2D or 3D patches where all voxel labels are zeros. Default: ``False``."
            },
            "filter_absent_class": {
                "type": "boolean",
                "$$description": [
                    "Discard 2D or 3D patches where all voxel labels are zero for one or more classes\n",
                    "(this is most relevant for multi-class models that need GT for all classes).\n",
                    "Default: ``False``."
                ]
            }
        }
    }


.. code-block:: JSON

    {
        "loader_parameters": {
            "patch_filter_params": {
                "filter_empty_mask": false,
                "filter_empty_input": false
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "roi_params",
        "description": "Parameters for the region of interest.",
        "type": "dict",
        "options": {
            "suffix": {
                "type": "string",
                "$$description": [
                    "Suffix of the derivative file containing the ROI used to crop\n",
                    "(e.g. ``_seg-manual``) with ``ROICrop`` as transform. Default: ``null``."
                ]
            },
            "slice_filter_roi": {
                "type": "int",
                "$$description": [
                    "If the ROI mask contains less than ``slice_filter_roi`` non-zero voxels\n",
                    "the slice will be discarded from the dataset. This feature helps with\n",
                    "noisy labels, e.g., if a slice contains only 2-3 labeled voxels, we do\n",
                    "not want to use these labels to crop the image. This parameter is only\n",
                    "considered when using ``ROICrop``. Default: ``null``."
                ]
            }
        }
    }



.. code-block:: JSON

    {
        "loader_parameters": {
            "roi_params": {
                "suffix": null,
                "slice_filter_roi": null
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "soft_gt",
        "$$description": [
            "Indicates if a soft mask will be used as ground-truth to train\n",
            "and / or evaluate a model. In particular, the masks are not binarized\n",
            "after interpolations implied by preprocessing or data-augmentation operations.\n",
            "Approach inspired by the `SoftSeg <https://arxiv.org/ftp/arxiv/papers/2011/2011.09041.pdf>`__ paper. Default: ``False``."
        ],
        "type": "boolean"
    }

.. code-block:: JSON

    {
        "loader_parameters": {
            "soft_gt": true
        }
    }

.. note::
    To get the full advantage of the soft segmentations, in addition to setting
    ``soft_gt: true`` the following keys in the config file must also be changed:
    (i) ``final_activation: relu`` - to use the normalized ReLU activation function
    (ii) ``loss: AdapWingLoss`` - a regression loss described in the
    paper. Note: It is also recommended to use the ``DiceLoss`` since convergence
    with ``AdapWingLoss`` is sometimes difficult to achieve.

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "is_input_dropout",
        "$$description": [
            "Indicates if input-level dropout should be applied during training.\n",
            "This option trains a model to be robust to missing modalities by setting \n",
            "to zero input channels (from 0 to all channels - 1). Always at least one \n",
            "channel will remain. If one or more modalities are already missing, they will \n",
            "be considered as dropped. Default: ``False``."
        ],
        "type": "boolean"
    }

.. code-block:: JSON

    {
        "loader_parameters": {
            "is_input_dropout": true
        }
    }



Split Dataset
-------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "fname_split",
        "$$description": [
            "Name of the `joblib <https://joblib.readthedocs.io/en/latest/>`__ file that was generated during a previous training, and that contains the list of training/validation/testing filenames.\n",
            "Specifying the .joblib file ensures reproducible data splitting across multiple trainings. When specified, the other split parameters are\n", 
            "ignored. If ``null`` is specified, a new splitting scheme is performed."
        ],
        "type": "string"
    }


.. code-block:: JSON

    {
        "split_dataset": {
            "fname_split": null
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "random_seed",
        "$$description": [
            "Seed used by the random number generator to split the dataset between\n",
            "training/validation/testing sets. The use of the same seed ensures the same split between\n",
            "the sub-datasets, which is useful for reproducibility. Default: ``6``."
        ],
        "type": "int"
    }

.. code-block:: JSON

    {
        "split_dataset": {
            "random_seed": 6
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "control_randomness",
        "description": [
            "Indicates if some sources of randomness in the experiments should be reduced by setting random_seed for numpy and torch. Default: ``False``.",
        "type": "boolean"
    }


.. code-block:: JSON

    {
        "split_dataset": {
            "control_randomness": true
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "split_method",
        "$$description": [
            "Metadata contained in a BIDS tabular (TSV) file or a BIDS sidecar JSON file on which the files are shuffled\n",
            "then split between train/validation/test, according to ``train_fraction`` and ``test_fraction``.\n",
            "For examples, ``participant_id`` will shuffle all participants from the ``participants.tsv`` file\n",
            "then split between train/validation/test sets. Default: ``participant_id``."
        ],
        "type": "string"
    }

.. code-block:: JSON

    {
        "split_dataset": {
            "split_method": "participant_id"
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "data_testing",
        "$$description": ["(Optional) Used to specify a custom metadata to only include in the testing dataset (not validation).\n",
            "For example, to not mix participants from different institutions between the train/validation set and the test set,\n",
            "use the column ``institution_id`` from ``participants.tsv`` in ``data_type``.\n"
        ],
        "type": "dict",
        "options": {
            "data_type": {
                "$$description": [
                    "Metadata to include in the testing dataset.\n",
                    "If specified, the ``test_fraction`` is applied to this metadata.\n",
                    "If not specified, ``data_type`` is the same as ``split_method``. Default: ``null``."
                ],
                "type": "string"
            },
            "data_value": {
                "$$description": [
                    "(Optional) List of metadata values from the ``data_type`` column to include in the testing dataset.\n",
                    "If specified, the testing set contains only files from the ``data_value`` list and the ``test_fraction`` is not used.\n",
                    "If not specified, create a random ``data_value`` according to ``data_type`` and ``test_fraction``. Default: ``[]``."
                ],
                "type": "list"
            }
        }
    }

.. code-block:: JSON

    {
        "split_dataset": {
            "data_testing": {"data_type": "institution_id", "data_value":[]}
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "balance",
        "$$description": [
            "Metadata contained in ``participants.tsv`` file with categorical values. Each category\n",
            "will be evenly distributed in the training, validation and testing datasets. Default: ``null``."
        ],
        "type": "string",
        "required": "false"
    }

.. code-block:: JSON

    {
        "split_dataset": {
            "balance": null
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "train_fraction",
        "description": "Fraction of the dataset used as training set. Default: ``0.6``.",
        "type": "float",
        "range": "[0, 1]"
    }

.. code-block:: JSON

    {
        "split_dataset": {
            "train_fraction": 0.6
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "test_fraction",
        "description": "Fraction of the dataset used as testing set. Default: ``0.2``.",
        "type": "float",
        "range": "[0, 1]"
    }

.. code-block:: JSON

    {
        "split_dataset": {
            "test_fraction": 0.2
        }
    }

.. note::
    .. line-block::
            The fraction of the dataset used as validation set will correspond to ``1 - train_fraction - test_fraction``.
            For example: ``1 - 0.6 - 0.2 = 0.2``.


Cascaded Models
---------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "object_detection_params",
        "type": "dict",
        "required": "false",
        "options": {
            "object_detection_path": {
                "type": "string",
                "$$description": [
                    "Path to the object detection model. The folder,\n",
                    "configuration file, and model need to have the same name\n",
                    "(e.g. ``findcord_tumor/``, ``findcord_tumor/findcord_tumor.json``, and\n",
                    "``findcord_tumor/findcord_tumor.onnx``, respectively). The model's prediction\n",
                    "will be used to generate bounding boxes. Default: ``null``."
                ]
            },
            "safety_factor": {
                "type": "[int, int, int]",
                "$$description": [
                    "List of length 3 containing the factors to multiply each dimension of the\n",
                    "bounding box. Ex: If the original bounding box has a size of 10x20x30 with\n",
                    "a safety factor of [1.5, 1.5, 1.5], the final dimensions of the bounding box\n",
                    "will be 15x30x45 with an unchanged center. Default: ``[1.0, 1.0, 1.0]``."
                ]
            }
       }
   }

.. code-block:: JSON

    {
        "object_detection_params": {
            "object_detection_path": null,
            "safety_factor": [1.0, 1.0, 1.0]
        }
    }



Training Parameters
-------------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "batch_size",
        "type": "int",
        "description": "Defines the number of samples that will be propagated through the network. Default: ``18``.",
        "range": "(0, inf)"
    }

.. code-block:: JSON

    {
        "training_parameters": {
            "batch_size": 24
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "loss",
        "$$description": [
            "Metadata for the loss function. Other parameters that could be needed in the\n",
            "Loss function definition: see attributes of the Loss function of interest (e.g. ``'gamma': 0.5`` for ``FocalLoss``)."
        ],
        "type": "dict",
        "options": {
            "name": {
                "type": "string",
                "description": "Name of the loss function class. See :mod:`ivadomed.losses`. Default: ``DiceLoss``."
            }
        }
    }

.. code-block:: JSON

    {
        "training_parameters": {
            "loss": {
                "name": "DiceLoss"
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "training_time",
        "type": "dict",
        "options": {
            "num_epochs": {
                "type": "int",
                "range": "(0, inf)"
            },
            "early_stopping_epsilon": {
                "type": "float",
                "$$description": [
                    "If the validation loss difference during one epoch\n",
                    "(i.e. ``abs(validation_loss[n] - validation_loss[n-1]`` where n is the current epoch)\n",
                    "is inferior to this epsilon for ``early_stopping_patience`` consecutive epochs,\n",
                    "then training stops. Default: ``0.001``."
                ]
            },
            "early_stopping_patience": {
                "type": "int",
                "range": "(0, inf)",
                "$$description": [
                    "Number of epochs after which the training is stopped if the validation loss\n",
                    "improvement is smaller than ``early_stopping_epsilon``. Default: ``50``."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "training_parameters": {
            "training_time": {
                "num_epochs": 100,
                "early_stopping_patience": 50,
                "early_stopping_epsilon": 0.001
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "scheduler",
        "type": "dict",
        "description": "A predefined framework that adjusts the learning rate between epochs or iterations as the training progresses.",
        "options": {
            "initial_lr": {
                "type": "float",
                "description": "Initial learning rate. Default: ``0.001``."
            },
            "lr_scheduler": {
                "type": "dict",
                "options": {
                    "name": {
                        "type": "string",
                        "$$description": [
                            "One of ``CosineAnnealingLR``, ``CosineAnnealingWarmRestarts`` and ``CyclicLR``.\n",
                            "Please find documentation `here <https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate>`__.",
                            "Default: ``CosineAnnealingLR``."
                        ]
                    },
                    "max_lr": {
                        "type": "float",
                        "description": "Upper learning rate boundaries in the cycle for each parameter group. Default: ``1e-2``."
                    },
                    "base_lr": {
                        "type": "float",
                        "$$description": [
                            "Initial learning rate which is the lower boundary in the cycle for each parameter group.\n",
                            "Default: ``1e-5``."
                        ]
                    }
                },
                "description": "Other parameters depend on the scheduler of interest."
            }
        }
    }

.. code-block:: JSON

    {
        "training_parameters": {
            "scheduler": {
                "initial_lr": 0.001,
                "lr_scheduler": {
                    "name": "CosineAnnealingLR",
                    "max_lr": 1e-2,
                    "base_lr": 1e-5
                }
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "balance_samples",
        "description": "Balance labels in both the training and the validation datasets.",
        "type": "dict",
        "options": {
          "applied": {
              "type": "boolean",
              "description": "Indicates whether to use a balanced sampler or not. Default: ``False``."
          },
          "type": {
              "type": "string",
              "$$description": [
                "Indicates which metadata to use to balance the sampler.\n",
                "Choices: ``gt`` or  the name of a column from the ``participants.tsv`` file\n",
                "(i.e. subject-based metadata). Default: ``gt``."
              ]
          }
        }
     }

.. code-block:: JSON

    {
        "training_parameters": {
            "balance_samples": {
                "applied": false,
                "type": "gt"
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "mixup_alpha",
        "description": "Alpha parameter of the Beta distribution, see `original paper on
        the Mixup technique <https://arxiv.org/abs/1710.09412>`__. Default: ``null``.",
        "type": "float"
    }

.. code-block:: JSON

    {
        "training_parameters": {
            "mixup_alpha": null
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "transfer_learning",
        "type": "dict",
        "$$description": ["A learning method where a model pretrained for a task is reused as the starting point",
            "for a model on a second task."
        ],
        "options": {
            "retrain_model": {
                "type": "string",
                "$$description": [
                    "Filename of the pretrained model (``path/to/pretrained-model``). If ``null``,\n",
                    "no transfer learning is performed and the network is trained from scratch. Default: ``null``."
                ]
            },
            "retrain_fraction": {
                "type": "float",
                "range": "[0, 1]",
                "$$description": [
                    "Controls the fraction of the pre-trained model that will be fine-tuned. For\n",
                    "instance, if set to 0.5, the second half of the model will be fine-tuned while\n",
                    "the first layers will be frozen. Default: ``1.0``."
                ]
            },
            "reset": {
                "type": "boolean",
                "$$description": ["If true, the weights of the layers that are not frozen are reset.",
                    "If false, they are kept as loaded. Default: ``True``."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "training_parameters": {
            "transfer_learning": {
                "retrain_model": null,
                "retrain_fraction": 1.0,
                "reset": true
            }
        }
   }


Architecture
------------

Architectures for both segmentation and classification are available and
described in the :ref:`architectures` section. If the selected architecture is listed in the
`loader <https://github.com/ivadomed/ivadomed/blob/lr/fixing_documentation/ivadomed/loader/loader.py>`__ file, a
classification (not segmentation) task is run. In the case of a
classification task, the ground truth will correspond to a single label
value extracted from ``target``, instead being an array (the latter
being used for the segmentation task).


.. jsonschema::

   {
       "$schema": "http://json-schema.org/draft-04/schema#",
       "title": "default_model",
       "required": "true",
       "type": "dict",
       "$$description": [
           "Define the default model (``Unet``) and mandatory parameters that are common to all available :ref:`architectures`.\n",
           "For custom architectures (see below), the default parameters are merged with the parameters that are specific\n",
           "to the tailored architecture."
       ],
       "options": {
           "name": {
               "type": "string",
               "description": "Default: ``Unet``"
           },
           "dropout_rate": {
               "type": "float",
               "description": "Default: ``0.3``"
           },
           "bn_momentum": {
               "type": "float",
               "$$description": [
                    "Defines the importance of the running average: (1 - `bn_momentum`). A large running\n",
                    "average factor will lead to a slow and smooth learning.\n",
                    "See `PyTorch's BatchNorm classes for more details. <https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html>`__ for more details. Default: ``0.1``\n"
               ]

           },
           "depth": {
               "type": "int",
               "range": "(0, inf)",
               "description": "Number of down-sampling operations. Default: ``3``"
           },
           "final_activation": {
               "type": "string",
               "required": "false",
               "$$description": [
                   "Final activation layer. Options: ``sigmoid`` (default), ``relu`` (normalized ReLU), or ``softmax``."
               ]
           },
           "length_2D": {
                "type": "[int, int]",
                "description": "(Optional) Size of the 2D patches used as model's input tensors.",
                "required": "false"
            },
            "stride_2D": {
                "type": "[int, int]",
                "$$description": [
                    "(Optional) Strictly positive integers: Pixels' shift over the input matrix to create 2D patches.\n",
                    "Ex: Stride of [1, 2] will cause a patch translation of 1 pixel in the 1st dimension and 2 pixels in\n",
                    "the 2nd dimension at every iteration until the whole input matrix is covered."
                ],
                "required": "false"
            },
           "is_2d": {
               "type": "boolean",
               "$$description": [
                   "Indicates if the model is 2D, if not the model is 3D. If ``is_2d`` is ``False``, then parameters\n",
                   "``length_3D`` and ``stride_3D`` for 3D loader need to be specified (see :ref:`Modified3DUNet <Modified3DUNet>`).\n",
                    "Default: ``True``."
               ]
           }
       }
   }


.. code-block:: JSON

    {
        "default_model": {
            "name": "Unet",
            "dropout_rate": 0.3,
            "bn_momentum": 0.1,
            "depth": 3,
            "final_activation": "sigmoid"
            "is_2d": true,
            "length_2D": [256, 256],
            "stride_2D": [244, 244]
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "FiLMedUnet",
        "type": "dict",
        "required": "false",
        "description": "U-Net network containing FiLM layers to condition the model with another data type (i.e. not an image).",
        "options": {
            "applied": {
                "type": "boolean",
                "description": "Set to ``true`` to use this model. Default: ``False``."
            },
            "metadata": {
                "type": "string",
                "options": {
                    "mri_params": {
                        "$$description": [
                            "Vectors of ``[FlipAngle, EchoTime, RepetitionTime, Manufacturer]``\n",
                            "(defined in the json of each image) are input to the FiLM generator."
                        ]
                    },
                    "contrasts": "Image contrasts (according to ``config/contrast_dct.json``) are input to the FiLM generator."
               },
               "$$description": [
                   "Choice between ``mri_params``, ``contrasts`` (i.e. image-based metadata) or the\n",
                   "name of a column from the participants.tsv file (i.e. subject-based metadata)."
               ]
            },
            "film_layers": {
                "description": "List of 0 or 1 indicating on which layer FiLM is applied."
            }
       }
   }

.. code-block:: JSON

    {
        "FiLMedUnet": {
            "applied": false,
            "metadata": "contrasts",
            "film_layers": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        }
    }


.. jsonschema::


    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "HeMISUnet",
        "type": "dict",
        "required": "false",
        "description": "A U-Net model inspired by HeMIS to deal with missing contrasts.",
        "options": {
            "applied": {
                "type": "boolean",
                "description": "Set to ``true`` to use this model."
            },
            "missing_probability": {
                "type": "float",
                "range": "[0, 1]",
                "$$description": [
                    "Initial probability of missing image contrasts as model's input\n",
                    "(e.g. 0.25 results in a quarter of the image contrasts, i.e. channels, that\n",
                    "will not be sent to the model for training)."
                ]
            },
            "missing_probability_growth": {
                "type": "float",
                "$$description": [
                    "Controls missing probability growth at each epoch: at each epoch, the\n",
                    "``missing_probability`` is modified with the exponent ``missing_probability_growth``.",
                ]
            }
         }
      }

.. code-block:: JSON

    {
        "HeMISUnet": {
            "applied": true,
            "missing_probability": 0.00001,
            "missing_probability_growth": 0.9,
            "contrasts": ["T1w", "T2w"],
            "ram": true,
            "path_hdf5": "/path/to/HeMIS.hdf5",
            "csv_path": "/path/to/HeMIS.csv",
            "target_lst": ["T2w"],
            "roi_lst": null
        }
    }

.. _Modified3DUNet:

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Modified3DUNet",
        "type": "dict",
        "required": "false",
        "$$description": [
            "The main differences with the original UNet resides in the use of LeakyReLU instead of ReLU, InstanceNormalisation\n",
            "instead of BatchNorm due to small batch size in 3D and the addition of segmentation layers in the decoder."
        ],
        "options": {
            "applied": {
                "type": "boolean",
                "description": "Set to ``true`` to use this model."
            },
            "length_3D": {
                "type": "[int, int, int]",
                "description": "Size of the 3D patches used as model's input tensors. Default: ``[128, 128, 16]``."
            },
            "stride_3D": {
                "type": "[int, int, int]",
                "$$description": [
                    "Voxels' shift over the input matrix to create patches. Ex: Stride of [1, 2, 3]\n",
                    "will cause a patch translation of 1 voxel in the 1st dimension, 2 voxels in\n",
                    "the 2nd dimension and 3 voxels in the 3rd dimension at every iteration until\n",
                    "the whole input matrix is covered. Default: ``[128, 128, 16]``."
                ]
            },
            "attention": {
                "type": "boolean",
                "description": "Use attention gates in the Unet's decoder. Default: ``False``.",
                "required": "false"
            },
            "n_filters": {
                "type": "int",
                "$$description": [
                    "Number of filters in the first convolution of the UNet.\n",
                    "This number of filters will be doubled at each convolution. Default: ``16``."
                ],
                "required": "false"
            }
       }
   }

.. code-block:: JSON

    {
        "Modified3DUNet": {
            "applied": false,
            "length_3D": [128, 128, 16],
            "stride_3D": [128, 128, 16],
            "attention": false,
            "n_filters": 8
        }
    }


Transformations
---------------

Transformations applied during data augmentation. Transformations are sorted in the order they are applied to the image samples. For each transformation, the following parameters are customizable:

- ``applied_to``: list between ``"im", "gt", "roi"``. If not specified, then the transformation is applied to all loaded samples. Otherwise, only applied to the specified types: Example: ``["gt"]`` implies that this transformation is only applied to the ground-truth data.
- ``dataset_type``: list between ``"training", "validation", "testing"``. If not specified, then the transformation is applied to the three sub-datasets. Otherwise, only applied to the specified subdatasets. Example: ``["testing"]`` implies that this transformation is only applied to the testing sub-dataset.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "NumpyToTensor",
        "type": "dict",
        "description": "Converts nd array to tensor object."
    }

.. code-block:: JSON

    {
        "transformation": {
            "NumpyToTensor": {
                "applied_to": ["im", "gt"]
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "CenterCrop",
        "type": "dict",
        "$$description": [
            "Make a centered crop of a specified size."
        ],
        "options": {
            "size": {
                "type": "list[int]"
            },
            "applied_to": {
                "type": "list[str]"
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "CenterCrop": {
                "applied_to": ["im", "gt"],
                "size":  [512, 256, 16]
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "ROICrop",
        "type": "dict",
        "$$description": [
            "Make a crop of a specified size around a Region of Interest (ROI).",
        ],
        "options": {
            "size": {
                "type": "list[int]"
            },
            "applied_to": {
                "type": "list[str]"
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "ROICrop": {
                "size": [48, 48],
                "applied_to": ["im", "roi"]
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "NormalizeInstance",
        "type": "dict",
        "$$description": [
            "Normalize a tensor or an array image with mean and standard deviation estimated from\n",
            "the sample itself."
        ],
        "options": {
            "applied_to": {
                "type": "list[str]"
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "NormalizeInstance": {
                "applied_to": ["im"]
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "RandomAffine",
        "type": "dict",
        "description": "Apply Random Affine transformation.",
        "options": {
            "degrees": {
                "type": "float or tuple(float)",
                "range": "(0, inf)",
                "$$description": [
                    "Positive float or list (or tuple) of length two. Angles in degrees. If only\n",
                    "a float is provided, then rotation angle is selected within the range\n",
                    "[-degrees, degrees]. Otherwise, the tuple defines this range."
                ]
            },
            "translate": {
                "type": "list[float]",
                "range": "[0, 1]",
                "$$description": [
                    "Length 2 or 3 depending on the sample shape (2D or 3D). Defines\n",
                    "the maximum range of translation along each axis."
                ]
            },
            "scale": {
                "type": "list[float]",
                "range": "[0, 1]",
                "$$description": [
                    "Length 2 or 3 depending on the sample shape (2D or 3D). Defines\n",
                    "the maximum range of scaling along each axis. Default: ``[0., 0., 0.]``."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "RandomAffine": {
                "translate": [0.03, 0.03],
                "applied_to": ["im"],
                "dataset_type": ["training"],
                "scale": [0.1, 0.5],
                "degrees": 180
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "RandomShiftIntensity",
        "type": "dict",
        "description": "Add a random intensity offset.",
        "options": {
            "shift_range": {
                "type": "(float, float)",
                "description": "Range from which the offset applied is randomly selected."
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "RandomShiftIntensity": {
                "shift_range": [28.0, 30.0]
            }
        }
     }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "ElasticTransform",
        "type": "dict",
        "$$description": [
            "Applies elastic transformation. See also:\n",
            "`Best practices for convolutional neural networks
             applied to visual document analysis <http://cognitivemedium.com/assets/rmnist/Simard.pdf>`__."
        ],
        "options": {
            "alpha_range": {
                "type": "(float, float)",
                "description": "Deformation coefficient."
            },
            "sigma_range": {
                "type": "(float, float)",
                "description": "Standard deviation."
            },
            "p": {
                "type": "float",
                "description": "Probability. Default: ``0.1``"
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "ElasticTransform": {
                "alpha_range": [28.0, 30.0],
                "sigma_range":  [3.5, 4.5],
                "p": 0.1,
                "applied_to": ["im", "gt"],
                "dataset_type": ["training"]
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Resample",
        "type": "dict",
        "description": "Resample image to a given resolution.",
        "options": {
            "hspace": {
                "type": "float",
                "range": "[0, 1]",
                "description": "Resolution along the first axis, in mm."
            },
            "wspace": {
                "type": "float",
                "range": "[0, 1]",
                "description": "Resolution along the second axis, in mm."
            },
            "dspace": {
                "type": "float",
                "range": "[0, 1]",
                "description": "Resolution along the third axis, in mm."
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "Resample": {
                "hspace": 0.75,
                "wspace": 0.75,
                "dspace": 1
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "AdditiveGaussianNoise",
        "type": "dict",
        "description": "Adds Gaussian Noise to images.",
        "options": {
            "mean": {
                "type": "float",
                "description": "Mean of Gaussian noise. Default: ``0.0``."
            },
            "std": {
                "type": "float",
                "description": "Standard deviation of Gaussian noise. Default: ``0.01``."
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "AdditiveGaussianNoise": {
                "mean": 0.0,
                "std": 0.02
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "DilateGT",
        "type": "dict",
        "description": "Randomly dilate a ground-truth tensor.",
        "options": {
            "dilation_factor": {
                "type": "float",
                "$$description": [
                    "Controls the number of iterations of ground-truth dilation depending on\n",
                    "the size of each individual lesion, data augmentation of the training set.\n",
                    "Use ``0`` to disable."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "DilateGT": {
                "dilation_factor": 0
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "HistogramClipping",
        "description": "Perform intensity clipping based on percentiles.",
        "type": "dict",
        "options": {
            "min_percentile": {
                "type": "float",
                "range": "[0, 100]",
                "description": "Lower clipping limit. Default: ``5.0``."
            },
            "max_percentile": {
                "type": "float",
                "range": "[0, 100]",
                "description": "Higher clipping limit. Default: ``95.0``."
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "HistogramClipping": {
                "min_percentile": 50,
                "max_percentile": 75
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "Clahe",
        "type": "dict",
        "description": "Applies Contrast Limited Adaptive Histogram Equalization for enhancing the local image contrast.",
        "options": {
            "clip_limit": {
                "type": "float",
                "description": "Clipping limit, normalized between 0 and 1. Default: ``3.0``."
            },
            "kernel_size": {
                "type": "list[int]",
                "$$description": [
                    "Defines the shape of contextual regions used in the algorithm.\n",
                    "List length = dimension, i.e. 2D or 3D. Default: ``[8, 8]``."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "Clahe": {
                "clip_limit": 0.5,
                "kernel_size": [8, 8]
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "RandomReverse",
        "type": "dict",
        "description": "Make a randomized symmetric inversion of the different values of each dimensions."
    }

.. code-block:: JSON

    {
        "transformation": {
            "RandomReverse": {
                "applied_to": ["im"]
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "RandomGamma",
        "type": "dict",
        "$$description": [
            "Randomly changes the contrast of an image by gamma exponential."
        ],
        "options": {
            "log_gamma_range": {
                "type": "[float, float]",
                "description": "Log gamma range for changing contrast."
            },
            "p": {
                "type": "float",
                "description": "Probability of performing the gamma contrast. Default: ``0.5``."
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "RandomGamma": {
                "log_gamma_range": [-3.0, 3.0],
                "p": 0.5,
                "applied_to": ["im"],
                "dataset_type": ["training"]
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "RandomBiasField",
        "type": "dict",
        "$$description": [
            "Applies a random MRI bias field artifact to the image via ``torchio.RandomBiasField()``."
        ],
        "options": {
            "coefficients": {
                "type": "float",
                "description": "Maximum magnitude of polynomial coefficients."
            },
            "order": {
                "type": "int",
                "description": "Order of the basis polynomial functions."
            },
            "p": {
                "type": "float",
                "description": "Probability of applying the bias field. Default: ``0.5``."
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "RandomBiasField": {
                "coefficients": 0.5,
                "order": 3,
                "p": 0.5,
                "applied_to": ["im"],
                "dataset_type": ["training"]
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "RandomBlur",
        "type": "dict",
        "$$description": [
            "Applies a random blur to the image."
        ],
        "options": {
            "sigma_range": {
                "type": "(float, float)",
                "description": "Standard deviation range for the gaussian filter."
            },
            "p": {
                "type": "float",
                "description": "Probability of performing blur. Default: ``0.5``."
            }
        }
    }

.. code-block:: JSON

    {
        "transformation": {
            "RandomBlur": {
                "sigma_range": [0.0, 2.0],
                "p": 0.5,
                "applied_to": ["im"],
                "dataset_type": ["training"]
            }
        }
    }

.. _Uncertainty:

Uncertainty
-----------

Uncertainty computation is performed if ``n_it>0`` and at least
``epistemic`` or ``aleatoric`` is ``true``. Note: both ``epistemic`` and
``aleatoric`` can be ``true``.


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "epistemic",
        "type": "boolean",
        "description": "Model-based uncertainty with `Monte Carlo Dropout <https://arxiv.org/abs/1506.02142>`__. Default: ``false``."
    }

.. code-block:: JSON

    {
        "uncertainty": {
            "epistemic": true
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "aleatoric",
        "type": "boolean",
        "description": "Image-based uncertainty with `test-time augmentation <https://doi.org/10.1016/j.neucom.2019.01.103>`__. Default: ``false``."
    }

.. code-block:: JSON

    {
        "uncertainty": {
            "aleatoric": true
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "n_it",
        "type": "int",
        "description": "Number of Monte Carlo iterations. Set to 0 for no uncertainty computation. Default: ``0``."
    }

.. code-block:: JSON

    {
        "uncertainty": {
            "n_it": 2
        }
    }


Postprocessing
--------------

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "binarize_prediction",
        "type": "dict",
        "options": {
            "thr": {
                "type": "float",
                "range": "[0, 1]",
                "$$description": [
                    "Threshold. To use soft predictions (i.e. no binarisation, float between 0 and 1)\n",
                    "for metric computation, indicate -1. Default: ``0.5``."
                ]
            }
        },
        "$$description": [
            "Binarizes predictions according to the given threshold ``thr``. Predictions below the\n",
            "threshold become 0, and predictions above or equal to threshold become 1."
        ]
    }



.. code-block:: JSON

    {
        "postprocessing": {
            "binarize_prediction": {
                "thr": 0.1
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "binarize_maxpooling",
        "type": "dict",
        "$$description": [
            "Binarize by setting to 1 the voxel having the maximum prediction across all classes.\n",
            "Useful for multiclass models. No parameters required (i.e., {}). Default: ``{}``."
        ]
    }



.. code-block:: JSON

    {
        "postprocessing": {
            "binarize_maxpooling": {}
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "fill_holes",
        "type": "dict",
        "description": "Fill holes in the predictions. No parameters required (i.e., {}). Default: ``{}``."
    }



.. code-block:: JSON

    {
        "postprocessing": {
            "fill_holes": {}
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "keep_largest",
        "type": "dict",
        "$$description": [
            "Keeps only the largest connected object in prediction. Only nearest neighbors are\n",
            "connected to the center, diagonally-connected elements are not considered neighbors.\n",
            "No parameters required (i.e., {}). Default: ``{}``."
        ]
    }



.. code-block:: JSON

    {
        "postprocessing": {
            "keep_largest": {}
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "remove_noise",
        "type": "dict",
        "options": {
            "thr": {
                "type": "float",
                "range": "[0, 1]",
                "description": "Threshold. Threshold set to ``-1`` will not apply this postprocessing step. Default: ``-1``."
            }
        },
        "description": "Sets to zero prediction values strictly below the given threshold ``thr``."
    }



.. code-block:: JSON

    {
        "postprocessing": {
            "remove_noise": {
                "thr": 0.1
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "remove_small",
        "type": "dict",
        "$$description": [
            "Remove small objects from the prediction. An object is defined as a group of connected\n",
            "voxels. Only nearest neighbors are connected to the center, diagonally-connected\n",
            "elements are not considered neighbors."
        ],
        "options": {
            "thr": {
                "type": "int or list[int]",
                "$$description": [
                    "Minimal object size. If a list of thresholds is chosen, the length should\n",
                    "match the number of predicted classes. Default: ``3``."
                ]
            },
            "unit": {
                "type": "string",
                "$$description": [
                    "Either ``vox`` for voxels or ``mm3``. Indicates the unit used to define the\n",
                    "minimal object size. Default: ``vox``."
                ]
            }
        }
    }



.. code-block:: JSON

    {
        "postprocessing": {
            "remove_small": {
                "unit": "vox",
                "thr": 3
            }
        }
    }

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "threshold_uncertainty",
        "type": "dict",
        "$$description": [
            "Removes the most uncertain predictions (set to 0) according to a threshold ``thr``\n",
            "using the uncertainty file with the suffix ``suffix``. To apply this method,\n",
            "uncertainty needs to be evaluated on the predictions with the :ref:`uncertainty <Uncertainty>` parameter."
        ],
        "options": {
            "thr": {
                "type": "float",
                "range": "[0, 1]",
                "$$description": [
                    "Threshold. Threshold set to ``-1`` will not apply this postprocessing step."
                ]
            },
            "suffix": {
                "type": "string",
                "$$description": [
                    "Indicates the suffix of an uncertainty file. Choices: ``_unc-vox.nii.gz`` for\n",
                    "voxel-wise uncertainty, ``_unc-avgUnc.nii.gz`` for structure-wise uncertainty\n",
                    "derived from mean value of ``_unc-vox.nii.gz`` within a given connected object,\n",
                    "``_unc-cv.nii.gz`` for structure-wise uncertainty derived from coefficient of\n",
                    "variation, ``_unc-iou.nii.gz`` for structure-wise measure of uncertainty\n",
                    "derived from the Intersection-over-Union of the predictions, or ``_soft.nii.gz``\n",
                    "to threshold on the average of Monte Carlo iterations."
                ]
            }
        }
    }



.. code-block:: JSON

    {
        "postprocessing": {
            "threshold_uncertainty": {
                "thr": -1,
                "suffix": "_unc-vox.nii.gz"
            }
        }
    }


Evaluation Parameters
---------------------
Dict. Parameters to get object detection metrics (lesions true positive rate, lesions false detection rate
and Hausdorff score), and this, for defined object sizes.

.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "object_detection_metrics",
        "$$description": [
            "Indicate if object detection metrics (lesions true positive rate, lesions false detection rate\n",
            "and Hausdorff score) are computed or not at evaluation time. Default: ``true``",
        ],
        "type": "boolean"
    }

.. code-block:: JSON

    {
        "evaluation_parameters": {
            "object_detection_metrics": true
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "target_size",
        "type": "dict",
        "options": {
            "thr": {
                "type": "list[int]",
                "$$description": [
                    "These values will create several consecutive target size bins. For instance\n",
                    "with a list of two values, we will have three target size bins: minimal size\n",
                    "to first list element, first list element to second list element, and second\n",
                    "list element to infinity. Default: ``[20, 100]``.\n",
                    "``object_detection_metrics`` must be ``true`` for the target_size to apply."
                ]
            },
            "unit": {
                "type": "string",
                "$$description": [
                    "Either ``vox`` for voxels or ``mm3``. Indicates the unit used to define the\n",
                    "target object sizes. Default: ``vox``."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "evaluation_parameters": {
            "target_size": {
                "thr": [20, 100],
                "unit": "vox"
            }
        }
    }


.. jsonschema::

    {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "title": "overlap",
        "type": "dict",
        "options": {
            "thr": {
                "type": "int",
                "$$description": [
                    "Minimal object size overlapping to be considered a TP, FP, or FN. Default: ``3``.\n",
                    "``object_detection_metrics`` must be ``true`` for the overlap to apply."
                ]
            },
            "unit": {
                "type": "string",
                "$$description":[
                    "Either ``vox`` for voxels or ``mm3``. Indicates the unit used to define the overlap.\n",
                    "Default: ``vox``."
                ]
            }
        }
    }

.. code-block:: JSON

    {
        "evaluation_parameters": {
            "overlap": {
                "thr": 3,
                "unit": "vox"
            }
        }
    }
