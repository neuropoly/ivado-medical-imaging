.. |br| raw:: html

   <br />

Scripts
=======

This section contains a collection of useful scripts for quality control during
the training of models.

ivadomed_segment_image
""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.segment_image.segment_image

ivadomed_visualize_transforms
"""""""""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.visualize_transforms.run_visualization

ivadomed_convert_to_onnx
""""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.convert_to_onnx.convert_pytorch_to_onnx

ivadomed_automate_training
""""""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.automate_training.automate_training

.. autofunction:: ivadomed.scripts.automate_training.HyperparameterOption

.. autofunction:: ivadomed.scripts.automate_training.get_param_list

.. autofunction:: ivadomed.scripts.automate_training.make_config_list

.. autofunction:: ivadomed.scripts.automate_training.update_dict


ivadomed_compare_models
"""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.compare_models.compute_statistics

ivadomed_prepare_dataset_vertebral_labeling
"""""""""""""""""""""""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.prepare_dataset_vertebral_labeling.extract_mid_slice_and_convert_coordinates_to_heatmaps

ivadomed_extract_small_dataset
""""""""""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.extract_small_dataset.extract_small_dataset

ivadomed_training_curve
"""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.training_curve.run_plot_training_curves

ivadomed_download_data
""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.download_data.install_data

ivadomed_visualize_and_compare_testing_models
"""""""""""""""""""""""""""""""""""""""""""""

.. autofunction:: ivadomed.scripts.visualize_and_compare_testing_models.visualize_and_compare_models
