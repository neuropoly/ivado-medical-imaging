Data
====

Organization
------------

To facilitate the organization of data, ``ivadomed`` requires the data to be organized according to the
`Brain Imaging Data Structure <https://bids.neuroimaging.io/>`_ (BIDS) standard.
The details of the standard can be found in the `BIDS specification <https://bids-specification.readthedocs.io/>`_.

Validation
----------

The compliance of the dataset with BIDS can be validated with the `BIDS-validator
web version <http://bids-standard.github.io/bids-validator>`_.
Other options for validation are available `here <https://github.com/bids-standard/bids-validator/#quickstart>`_.

Examples
--------

An example of this organization is shown below for MRI data:

.. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/data/1920px-BIDS_Logo.png
    :alt: BIDS_Logo
    :width: 200

::

    dataset/
    └── dataset_description.json
    └── participants.tsv  <-------------------------------- Metadata describing subjects attributes e.g. sex, age, etc.
    └── sub-01  <------------------------------------------ Folder enclosing data for subject 1
    └── sub-02
    └── sub-03
        └── anat
            └── sub-03_T1w.nii.gz  <----------------------- MRI image in NIfTI format
            └── sub-03_T1w.json  <------------------------- Metadata including image parameters, MRI vendor, etc.
            └── sub-03_T2w.nii.gz
            └── sub-03_T2w.json
    └── derivatives
        └── labels
            └── sub-03
                └── anat
                    └── sub-03_seg-tumor-manual.nii.gz  <-- Manually-corrected segmentation
                    └── sub-03_seg-tumor-manual.json  <---- Metadata including author who performed the labeling and date

.. note:: For an exhaustive list of ``derivatives`` used in ``ivadomed``, please see our `wiki <https://github.com/ivadomed/ivadomed/wiki/repositories#derivatives>`_.

For usage in ``ivadomed``, additional examples are available in our tutorials, for `MRI data <https://ivadomed.org/tutorials/one_class_segmentation_2d_unet.html>`_ and `Microscopy data <https://ivadomed.org/tutorials/two_class_microscopy_seg_2d_unet.html>`_.
Further examples of the BIDS organization can be found in the
`BIDS-examples <https://github.com/bids-standard/bids-examples#dataset-index>`_ repository.
