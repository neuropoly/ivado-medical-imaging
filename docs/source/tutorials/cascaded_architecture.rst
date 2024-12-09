Cascaded architecture
=====================

    In this tutorial we will learn the following features:

    - Design a training scheme composed of two cascaded networks.
    - Visualize the training with tensorboard.
    - Generate a GIF to visualize the learning of the model.
    - Find the optimal threshold to binarize images based on the validation sub-dataset.

    In our example, the model will first locate the spinal cord (step 1). This localisation will then be used to crop the images around this region of interest, before segmenting the cerebrospinal fluid (CSF, step 2).

Download dataset
----------------

    A dataset example is available for this tutorial. If not already done, download the dataset with the following line.
    For more details on this dataset see :ref:`One-class segmentation with 2D U-Net<Download dataset>`.

    .. tabs::

        .. tab:: Command Line Interface

            .. code-block:: bash

                # Download data
                ivadomed_download_data -d data_example_spinegeneric

Configuration file
------------------

    In ``ivadomed``, training is orchestrated by a configuration file. Examples of configuration files are available in
    the ``ivadomed/config/`` and the documentation is available in :doc:`../configuration_file`.

    In this tutorial we will use the configuration file: ``ivadomed/config/config.json``.
    First off, copy this configuration file in your local directory to avoid modifying the source file:

    .. tabs::

        .. tab:: Command Line Interface

            .. code-block:: bash

               cp <PATH_TO_IVADOMED>/ivadomed/config/config.json .


    Then, open it with a text editor. Which you can `view directly here: <https://github.com/ivadomed/ivadomed/blob/master/ivadomed/config/config.json>`_ or you can see it in the collapsed JSON code block below.

        .. collapse:: Reveal the embedded config.json

            .. code-block:: json
                :linenos:

                {
                    "command": "train",
                    "gpu_ids": [0],
                    "path_output": "spineGeneric",
                    "model_name": "my_model",
                    "debugging": false,
                    "object_detection_params": {
                        "object_detection_path": null,
                        "safety_factor": [1.0, 1.0, 1.0]
                    },
                    "loader_parameters": {
                        "path_data": ["data_example_spinegeneric"],
                        "subject_selection": {"n": [], "metadata": [], "value": []},
                        "target_suffix": ["_seg-manual"],
                        "extensions": [".nii.gz"],
                        "roi_params": {
                            "suffix": null,
                            "slice_filter_roi": null
                        },
                        "contrast_params": {
                            "training_validation": ["T1w", "T2w", "T2star"],
                            "testing": ["T1w", "T2w", "T2star"],
                            "balance": {}
                        },
                        "slice_filter_params": {
                            "filter_empty_mask": false,
                            "filter_empty_input": true
                        },
                        "slice_axis": "axial",
                        "multichannel": false,
                        "soft_gt": false
                    },
                    "split_dataset": {
                        "fname_split": null,
                        "random_seed": 6,
                        "split_method" : "participant_id",
                        "data_testing": {"data_type": null, "data_value":[]},
                        "balance": null,
                        "train_fraction": 0.6,
                        "test_fraction": 0.2
                    },
                    "training_parameters": {
                        "batch_size": 18,
                        "loss": {
                            "name": "DiceLoss"
                        },
                        "training_time": {
                            "num_epochs": 100,
                            "early_stopping_patience": 50,
                            "early_stopping_epsilon": 0.001
                        },
                        "scheduler": {
                            "initial_lr": 0.001,
                            "lr_scheduler": {
                                "name": "CosineAnnealingLR",
                                "base_lr": 1e-5,
                                "max_lr": 1e-2
                            }
                        },
                        "balance_samples": {
                            "applied": false,
                            "type": "gt"
                        },
                        "mixup_alpha": null,
                        "transfer_learning": {
                            "retrain_model": null,
                            "retrain_fraction": 1.0,
                            "reset": true
                        }
                    },
                    "default_model": {
                        "name": "Unet",
                        "dropout_rate": 0.3,
                        "bn_momentum": 0.1,
                        "final_activation": "sigmoid",
                        "depth": 3
                    },
                    "FiLMedUnet": {
                        "applied": false,
                        "metadata": "contrasts",
                        "film_layers": [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
                    },
                    "Modified3DUNet": {
                        "applied": false,
                        "length_3D": [128, 128, 16],
                        "stride_3D": [128, 128, 16],
                        "attention": false,
                        "n_filters": 8
                    },
                    "uncertainty": {
                        "epistemic": false,
                        "aleatoric": false,
                        "n_it": 0
                    },
                    "postprocessing": {
                        "remove_noise": {"thr": -1},
                        "keep_largest": {},
                        "binarize_prediction": {"thr": 0.5},
                        "uncertainty": {"thr": -1, "suffix": "_unc-vox.nii.gz"},
                        "fill_holes": {},
                        "remove_small": {"unit": "vox", "thr": 3}
                    },
                    "evaluation_parameters": {
                        "target_size": {"unit": "vox", "thr": [20, 100]},
                        "overlap": {"unit": "vox", "thr": 3}
                    },
                    "transformation": {
                        "Resample":
                        {
                            "hspace": 0.75,
                            "wspace": 0.75,
                            "dspace": 1
                        },
                        "CenterCrop": {
                            "size": [128, 128]},
                        "RandomAffine": {
                            "degrees": 5,
                            "scale": [0.1, 0.1],
                            "translate": [0.03, 0.03],
                            "applied_to": ["im", "gt"],
                            "dataset_type": ["training"]
                        },
                        "ElasticTransform": {
                            "alpha_range": [28.0, 30.0],
                            "sigma_range":  [3.5, 4.5],
                            "p": 0.1,
                            "applied_to": ["im", "gt"],
                            "dataset_type": ["training"]
                        },
                      "NormalizeInstance": {"applied_to": ["im"]}
                    }
                }


    From this point onward, we will discuss some of the key parameters to use cascaded models. Most parameters are configurable only via modification of the configuration ``JSON file``.
    For those that supports commandline run time configuration, we included the respective command versions under the ``Command Line Interface`` tab

    At `this line <https://github.com/ivadomed/ivadomed/blob/76b36a0a0f7141feb2d5b00b33e4c3a06865fc2c/ivadomed/config/config.json#L6>`__ in the ``config.json`` is where you can update the ``debugging``.

    - ``debugging``: Boolean, create extended verbosity and intermediate outputs. Here we will look at the intermediate predictions
      with tensorboard, we therefore need to activate those intermediate outputs.

        .. tabs::

            .. tab:: JSON File

                  .. code-block:: json

                     "debugging": true

    At `this line <https://github.com/ivadomed/ivadomed/blob/76b36a0a0f7141feb2d5b00b33e4c3a06865fc2c/ivadomed/config/config.json#L8>`__ in the ``config.json`` is where you can update the ``object_detection_path`` within the ``object_detection_params`` sub-dictionary.

    - ``object_detection_params:object_detection_path``: Location of the object detection model. This parameter corresponds
      to the path of the first model from the cascaded architecture that segments the spinal cord. The packaged model in the
      downloaded dataset located in the folder `trained_model/seg_sc_t1-t2-t2s-mt` will be used to detect the spinal cord.
      This spinal cord segmentation model will be applied to the images and a bounding box will be created around this mask
      to crop the image.

        .. tabs::

            .. tab:: JSON File

                  .. code-block:: json

                     "object_detection_path": "<PATH_TO_DATASET>/trained_model/seg_sc_t1-t2-t2s-mt"

    At `this line <https://github.com/ivadomed/ivadomed/blob/76b36a0a0f7141feb2d5b00b33e4c3a06865fc2c/ivadomed/config/config.json#L9>`__ in the ``config.json`` is where you can update the ``safety_factor`` within the ``object_detection_params`` sub-dictionary.

    - ``object_detection_params:safety_factor``: Multiplicative factor to apply to each dimension of the bounding box. To
      ensure all the CSF is included, a safety factor should be applied to the bounding box generated from the spinal cord.
      A safety factor of 200% on each dimension is applied on the height and width of the image. The original depth of the
      bounding box is kept since the CSF should not be present past this border.

        .. tabs::

            .. tab:: JSON File

                  .. code-block:: json

                     "safety_factor": [2, 2, 1]

    At `this line <https://github.com/ivadomed/ivadomed/blob/76b36a0a0f7141feb2d5b00b33e4c3a06865fc2c/ivadomed/config/config.json#L14>`__ in the ``config.json`` is where you can update the ``target_suffix`` within the ``loader_parameters`` sub-dictionary.

    - ``loader_parameters:target_suffix``: Suffix of the ground truth segmentation. The ground truth are located under the
      ``DATASET/derivatives/labels`` folder. The suffix for CSF labels in this dataset is ``_csfseg-manual``:

        .. tabs::

            .. tab:: JSON File

                  .. code-block:: json

                     "target_suffix": ["_csfseg-manual"]

    At `this line <https://github.com/ivadomed/ivadomed/blob/76b36a0a0f7141feb2d5b00b33e4c3a06865fc2c/ivadomed/config/config.json#L20>`__ in the ``config.json`` is where you can update the ``contrast_params`` within the ``loader_parameters`` sub-dictionary.

    - ``loader_parameters:contrast_params``: Contrast(s) of interest. The CSF labels are only available in T2w contrast in
      the example dataset.

        .. tabs::

            .. tab:: JSON File

                  .. code-block:: json

                     "contrast_params": {
                         "training_validation": ["T2w"],
                         "testing": ["T2w"],
                         "balance": {}
                     }

    At `this line <https://github.com/ivadomed/ivadomed/blob/76b36a0a0f7141feb2d5b00b33e4c3a06865fc2c/ivadomed/config/config.json#L115>`__ in the ``config.json`` is where you can update the ``size`` within the ``transformation:CenterCrop`` sub-dictionary.

    - ``transformation:CenterCrop:size``: Crop size in voxel. Images will be cropped or padded to fit these dimensions. This
      allows all the images to have the same size during training. Since the images will be cropped around the spinal cord,
      the image size can be reduced to avoid large zero padding.

        .. tabs::

            .. tab:: JSON File

                  .. code-block:: json

                     "CenterCrop": {
                         "size": [64, 64]
                     }

Train model
-----------

    Once the configuration file is ready, run the training. ``ivadomed`` has an option to find a threshold value which optimized the dice score on the validation dataset. This threshold will be further used to binarize the predictions on testing data. Add the flag ``-t`` with an increment
    between 0 and 1 to perform this threshold optimization (i.e. ``-t 0.1`` will return the best threshold between 0.1,
    0.2, ..., 0.9)

    To help visualize the training, the flag ``--gif`` or ``-g`` can be used. The flag should be followed by the number of
    slices by epoch to visualize. For example, ``-g 2`` will generate 2 GIFs of 2 randomly selected slices from the
    validation set.

    Make sure to run the CLI command with the ``--train`` flag, and to point to the location of the dataset via the flag ``--path-data path/to/bids/data``.

    .. tabs::

        .. tab:: Command Line Interface

            .. code-block:: bash

               ivadomed --train -c config.json --path-data path/to/bids/data --path-output path/to/output/directory -t 0.01 -g 1


        .. tab:: JSON File

            If you prefer to use config files over CLI flags, set ``command`` to the following in you config file:

                .. code-block:: json

                   "command": "train"

            You can also set ``path_output``, and ``path_data`` arguments in your config file.

            Then run:

                .. tabs::

                    .. tab:: Command Line Interface

                        .. code-block:: bash

                            ivadomed -c config.json

    At the end of the training, the optimal threshold will be indicated:

    .. code-block:: console

       Running threshold analysis to find optimal threshold
        Optimal threshold: 0.01
        Saving plot: spineGeneric/roc.png


Visualize training data
-----------------------
    If the flag ``--gif`` or ``-g`` was used, the training can be visualized through gifs located in the folder specified by the ``--path-output`` flag
    ``<PATH_TO_OUT_DIR>/gifs``.

    .. figure:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/tutorials/cascaded_architecture/training.gif
       :width: 300
       :align: center

       Training visualization with GIF

    Another way to visualize the training is to use Tensorboard. Tensorboard helps to visualize the augmented input images,
    the model's prediction, the ground truth, the learning curves, and more. To access this data during or after training,
    use the following command-line:

    .. tabs::

            .. tab:: Command Line Interface

                .. code-block:: bash

                   tensorboard --logdir <PATH_TO_OUT_DIR>

    The following should be displayed in the terminal:

    .. code-block:: console

       Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
       TensorBoard 2.2.1 at http://localhost:6006/ (Press CTRL+C to quit)

    Open your browser and type the URL provided, in this case ``http://localhost:6006/``.
    In the scalars folder, the evolution of metrics, learning rate and loss through the epochs can be visualized.

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/tutorials/cascaded_architecture/tensorboard_scalar.png
       :align: center

    In the image folder, the training and validation ground truth, input images and predictions are displayed. With this
    feature, it is possible to visualize the cropping from the first model and confirm that the spinal cord
    was correctly located.

    .. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/tutorials/cascaded_architecture/tensorboard_images.png
       :align: center

Evaluate model
--------------
    - ``postprocessing:binarize_prediction``: Threshold at which predictions are binarized. Before testing the model,
      modify the binarization threshold to have a threshold adapted to the data:

    .. tabs::

        .. tab:: JSON File

              .. code-block:: json

                 "binarize_prediction": 0.01


    To test and apply this model on the testing dataset, go to the `Evaluate model` section of the tutorial
    :ref:`One-class segmentation with 2D U-Net<evaluate model>`.
