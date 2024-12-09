.. image:: https://raw.githubusercontent.com/ivadomed/doc-figures/main/index/overview_title.png
  :alt: Alternative text

|

``ivadomed`` is an integrated framework for medical image analysis with deep
learning, based on `PyTorch <https://pytorch.org/>`_. The name is a portmanteau between *IVADO* (The `Institute for data
valorization <https://ivado.ca/en/>`_) and *Medical*.

If you use ``ivadomed`` for your research, please cite `our paper <https://joss.theoj.org/papers/10.21105/joss.02868>`_:

.. code::

    @article{gros2021ivadomed,
      doi = {10.21105/joss.02868},
      url = {https://doi.org/10.21105/joss.02868},
      year = {2021},
      publisher = {The Open Journal},
      volume = {6},
      number = {58},
      pages = {2868},
      author = {Charley Gros and Andreanne Lemay and Olivier Vincent and Lucas Rouhier and Marie-Helene Bourget and Anthime Bucquet and Joseph Paul Cohen and Julien Cohen-Adad},
      title = {ivadomed: A Medical Imaging Deep Learning Toolbox},
      journal = {Journal of Open Source Software}
    }

Home
====

.. toctree::
   :maxdepth: 1
   :caption: Overview

   purpose.rst
   technical_features.rst
   use_cases.rst

.. toctree::
   :maxdepth: 1
   :caption: Getting started

   installation.rst
   data.rst
   configuration_file.rst
   usage.rst
   architectures.rst
   pretrained_models.rst
   scripts.rst
   help.rst

.. _tutorials:
.. toctree::
   :maxdepth: 1
   :caption: Tutorials

   tutorials/one_class_segmentation_2d_unet.rst
   tutorials/cascaded_architecture.rst
   tutorials/uncertainty.rst
   tutorials/automate_training.rst
   tutorials/two_class_microscopy_seg_2d_unet.rst

.. toctree::
   :maxdepth: 1
   :caption: Developer section

   contributing.rst
   api_ref.rst
   contributors.rst
   license.rst
