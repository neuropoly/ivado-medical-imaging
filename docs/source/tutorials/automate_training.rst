Optimize Hyperparameters
========================

For training, you may want to determine which hyperparameters will be the best to use. To do this,
we have the function ``ivadomed_automate_training``.


Step 1: Download Example Data
-----------------------------

To download the dataset (~490MB), run the following command in your terminal:

.. code-block:: bash

   ivadomed_download_data -d data_example_spinegeneric


Step 2: Copy Configuration File
-------------------------------

In ``ivadomed``, training is orchestrated by a configuration file. Examples of configuration files
are available in the ``ivadomed/config/`` and the documentation is available in :doc:`../configuration_file`.

In this tutorial we will use the configuration file: ``ivadomed/config/config.json``.
Copy this configuration file in your local directory (to avoid modifying the source file):

.. code-block:: bash

   cp <PATH_TO_IVADOMED>/ivadomed/config/config.json .

Then, open it with a text editor. Below we will discuss some of the key parameters to perform a one-class 2D
segmentation training.


Step 3: Create Hyperparameters Config File
------------------------------------------

The hyperparameter config file should have the same layout as the config file. To select
a hyperparameter you would like to vary, just list the different options under the
appropriate key.

In our example, we have 3 hyperparameters we would like to vary: ``batch_size``, ``loss``, and
``depth``. In your directory, create a new file called: ``config_hyper.json``. Open this
in a text editor and add the following:

.. code-block:: JSON

    {
        "training_parameters": {
            "batch_size": [2, 64],
            "loss": [
                {"name": "DiceLoss"},
                {"name": "FocalLoss", "gamma": 0.2, "alpha" : 0.5}
            ]
        },
        "default_model": {"depth": [2, 3, 4]}
    }

Step 4: (Optional) Change the Training Epochs
---------------------------------------------

The default number of training epochs in the ``config.json`` file is ``100``; however,
depending on your computer, this could be quite slow (especially if you don't have any GPUs).

To change the number of epochs, open the ``config.json`` file and change the following:

.. code-block:: JSON

    {
        "training_parameters": {
            "training_time": {
                "num_epochs": 1
            }
        }
    }


Step 5: Run the Code
--------------------

Default
^^^^^^^

If neither ``all-combin`` nor ``multi-params`` is selected, then the hyperparameters will be
combined as follows into a ``config_list``.

.. note::

    I am not showing the actual ``config_list`` here as it would take up too much space. The options
    listed below are incorporated into the base config file in ``config.json``.

.. code-block::

    batch_size = 2, loss = "DiceLoss", depth = 3
    batch_size = 64, loss = "DiceLoss", depth = 3
    batch_size = 18, loss = "FocalLoss", depth = 3
    batch_size = 18, loss = "DiceLoss", depth = 2
    batch_size = 18, loss = "DiceLoss", depth = 3
    batch_size = 18, loss = "DiceLoss", depth = 4

To run this:

.. code-block:: bash

    ivadomed_automate_training -c config.json -ch config_hyper.json \
    -n 1

All Combinations
^^^^^^^^^^^^^^^^

If the flag ``all-combin`` is selected, the hyperparameter options will be combined
combinatorically.

.. code-block::

    batch_size = 2, loss = "DiceLoss", depth = 2
    batch_size = 2, loss = "FocalLoss", depth = 2
    batch_size = 2, loss = "DiceLoss", depth = 3
    batch_size = 2, loss = "FocalLoss", depth = 3
    batch_size = 2, loss = "DiceLoss", depth = 4
    batch_size = 2, loss = "FocalLoss", depth = 4
    batch_size = 2, loss = "DiceLoss", depth = 4
    batch_size = 2, loss = "FocalLoss", depth = 4
    batch_size = 64, loss = "DiceLoss", depth = 2
    batch_size = 64, loss = "FocalLoss", depth = 2
    batch_size = 64, loss = "DiceLoss", depth = 3
    batch_size = 64, loss = "FocalLoss", depth = 3
    batch_size = 64, loss = "DiceLoss", depth = 4
    batch_size = 64, loss = "FocalLoss", depth = 4
    batch_size = 64, loss = "DiceLoss", depth = 4
    batch_size = 64, loss = "FocalLoss", depth = 4

To run:

.. code-block:: bash

    ivadomed_automate_training -c config.json -ch config_hyper.json \
    -n 1 --all-combin

Multiple Parameters
^^^^^^^^^^^^^^^^^^^

If the flag ``multi-params`` is selected, the elements from each hyperparameter list will be
selected sequentially, so all the first elements, then all the second elements, etc. If the lists
are different lengths, say ``len(list_a) = n`` and ``len(list_b) = n+m``, where ``n`` and ``m``
are strictly positive integers, then we will only use the first ``n`` elements.

.. code-block::

    batch_size = 2, loss = "DiceLoss", depth = 2
    batch_size = 64, loss = "FocalLoss", depth = 3

To run:

.. code-block:: bash

    ivadomed_automate_training -c config.json -ch config_hyper.json \
    -n 1 --multi-params

Step 6: Results
---------------

There will be an output file called ``detailed_results.csv``. This file gives an overview of the
results from all the different trials. For a more fine-grained analysis, you can also look
at each of the log directories (there is one for each config option).

An example of the ``detailed_results.csv``:

.. csv-table::
   :file: detailed_results.csv
