Usage
=====

.. _usage:

Command line tools
------------------

New model can be generated using the command-line tool from the
terminal:

::

    ivadomed [command] -c path/to/config_file.json --path-data path/to/bids/data --path-output path/to/output/directory

``[command]`` represents the following choice of flags:

    ``--train``: train a model on a training/validation sub-datasets

    ``--test``: evaluate a trained model on a testing sub-dataset

    ``--segment``: segment a entire dataset using a trained model. Note that you may only specify one command flag at a time.

Note that the command CLI flag is optional and can be specified instead via the configuration file (see :ref:`configuration_file:Configuration File` ).
If not set via CLI, then you MUST specify this field in the configuration file.

``config_file.json`` is a configuration file, which parameters are
described in the :ref:`configuration_file:Configuration File`. This flag is *required*.

``path/to/bids/data`` is the location of the dataset. As discussed in :doc:`Data <data>`, the dataset
should conform to the BIDS standard. Modify the path so it points to the location of the downloaded dataset.

``path/to/output/directory`` is the folder name that will contain the output files (e.g., trained model, predictions, results)

Note that both path CLI flags are optional and can be specified instead via the configuration file.
If not set via CLI, then you MUST specify this field in the configuration file.

Please see section ``TUTORIALS`` to run this command on an example dataset.

Additional optional flags with ``--segment`` command for models trained with 2D patches (not available for 3D models):

    ``--no-patch``: 2D patches are not used while segmenting with models trained with patches. The ``--no-patch`` flag supersedes the
    ``--overlap-2d`` flag. This option may not be suitable with large images depending on computer RAM capacity.

    ``--overlap-2d``: Custom overlap for 2D patches while segmenting. Example: ``--overlap-2d 48 48`` for an overlap of 48 pixels between patches in X and Y respectively. Default model overlap is used otherwise.
