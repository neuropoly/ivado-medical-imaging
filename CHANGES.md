 ## v2.9.10 (2024-03-12)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.9...release)

**BUG**

 - Fix edge case bug where grayscale image has an alpha channel, minor rtd fix.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1313)

**DEPENDENCIES**

 - Test upgrades for pinned dependencies to improve downstream compatibility.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1308)

 ## v2.9.9 (2023-12-11)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.8...release)

**FEATURE**
 - Introduce `segment_image` CLI. [View pull request](https://github.com/ivadomed/ivadomed/pull/1254)

**CI**
 - chore: remove ubuntu-18.04 and python 3.7 from `run_tests` workflow. [View pull request](https://github.com/ivadomed/ivadomed/pull/1298)

**BUG**
 - chore: Dependency Maintenance (imageio 2->3, pyBIDS<0.15.6, readthedocs.yml v2, Python 3.7->3.8). [View pull request](https://github.com/ivadomed/ivadomed/pull/1297)
 - Fix NormalizeInstance for uniform samples. [View pull request](https://github.com/ivadomed/ivadomed/pull/1267)

**INSTALLATION**
 - Test out newer versions of PyTorch (`torch>=2.0.0`) for compatibility with downstream projects (SCT, ADS). [View pull request](https://github.com/ivadomed/ivadomed/pull/1304)

**DOCUMENTATION**
 - Update doc for segment_image.py. [View pull request](https://github.com/ivadomed/ivadomed/pull/1290)
 - Clarified usage for `fname_split`. [View pull request](https://github.com/ivadomed/ivadomed/pull/1283)
 - Point to the URL specific to the learning rate. [View pull request](https://github.com/ivadomed/ivadomed/pull/1281)
 - Match location of `object_detection` in the template config file and in the documentation. [View pull request](https://github.com/ivadomed/ivadomed/pull/1278)
 - Resolve the discrepancies. [View pull request](https://github.com/ivadomed/ivadomed/pull/1271)
 - Clarify usage of --no-patch for 2D only. [View pull request](https://github.com/ivadomed/ivadomed/pull/1265)

**ENHANCEMENT**
 - Do not use wandb if not specified in the config file. [View pull request](https://github.com/ivadomed/ivadomed/pull/1253)

**REFACTORING**
 - Harmonize get_item in MRI2D dataset and MRI3D dataset. [View pull request](https://github.com/ivadomed/ivadomed/pull/1266)

 ## v2.9.8 (2023-01-04)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.7...release)

**CI**

 - chore: remove numpy related deprecations in support of v1.24.0 .  [View pull request](https://github.com/ivadomed/ivadomed/pull/1246)

**BUG**

 - Fix 3D training with data augmentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1222)
 - Fix GPU behavior in segment_volume.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1209)

**DOCUMENTATION**

 - Clarify testing output files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1244)
 - doc: clarify validation fraction.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1207)

**DEPENDENCIES**

 - chore: remove numpy related deprecations in support of v1.24.0 .  [View pull request](https://github.com/ivadomed/ivadomed/pull/1246)

**ENHANCEMENT**

 - fix/feat: update the path for wandb logs.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1237)

**REFACTORING**

 - Clarify testing output files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1244)
 - Remove force indexing of microscopy and update ct.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1203)
 - Refactoring in ConfigurationManager class (config_manager.py).  [View pull request](https://github.com/ivadomed/ivadomed/pull/1195)
 - Type Hint for utils.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1162)
 - Type Hint for slice_filter.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1161)
 - Type Hint for segmentation_pair.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1160)
 - Type Hint for sample_meta_data.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1159)
 - Type Hint for patch_filter.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1158)
 - Type Hint for mri3d_subvolume_segmentation_dataset.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1157)
 - Type Hint for mri2d_segmentation_dataset.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1156)
 - type hinting for bids_dataset.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1155)
 - Type Hint for bids_dataframe.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1154)
 - Type Hint for bids3d_dataset.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1153)
 - Type Hinting for balanced_sampler.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1152)
 - Typehint for loader/film.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1151)
 - Type Hinting for loader.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1150)

 ## v2.9.7 (2022-10-31)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.6...release)

**FEATURE**

 - feat: update default args for `wandb.login()`.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1193)
 - Add Config Parameter to Disable Validation When Loading BIDS Info.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1168)
 - Add auto disk cache capability to mri2d and mri3d dataset classes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1121)
 - Segment 2D images without patches.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1101)

**CI**

 - chore: add an upper bound version specifier for pandas to prevent breaking the tests.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1194)
 - Drop testing on macOS 10.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1190)
 - Only run tests on code changes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1186)
 - chore: upgrade run_tests workflow.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1146)
 - Don't install ivadomed just to run pre-commit checks..  [View pull request](https://github.com/ivadomed/ivadomed/pull/1145)

**BUG**

 - chore: add an upper bound version specifier for pandas to prevent breaking the tests.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1194)
 - Drop testing on macOS 10.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1190)
 - Resolve imageio v2->v3 imread deprecation warnings.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1181)
 - fix: adapt the filenames of the predictions in pred_masks as per target_suffix.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1173)
 - Fix long evaluation time on microscopy images.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1081)

**INSTALLATION**

 - chore: update installation for ivadomed tutorials.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1200)
 - Remove deprecated torch and dev installation instructions.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1179)
 - Get Rid of Python 3.6.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1149)
 - Support python3.10.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1137)

**DOCUMENTATION**

 - fix: update link to contribution guidelines.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1196)
 - Tests README: Add instructions to install testing-related packages.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1180)
 - Remove deprecated torch and dev installation instructions.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1179)
 - Add description and default values to some parameters..  [View pull request](https://github.com/ivadomed/ivadomed/pull/1174)
 - Add Note in WandB.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1171)

**DEPENDENCIES**

 - chore: add an upper bound version specifier for pandas to prevent breaking the tests.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1194)
 - Support python3.10.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1137)

**ENHANCEMENT**

 - feat: update default args for `wandb.login()`.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1193)
 - Only run tests on code changes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1186)
 - Transformation on Subvolume for mri3d_subvolume_segmentation_dataset.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1169)
 - Support python3.10.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1137)
 - Add syntax highlighting and improve flow of the Colab tutorials.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1127)
 - Add auto disk cache capability to mri2d and mri3d dataset classes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1121)

**TESTING**

 - Drop testing on macOS 10.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1190)
 - Only run tests on code changes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1186)
 - Add syntax highlighting and improve flow of the Colab tutorials.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1127)

**REFACTORING**

 - feat: update default args for `wandb.login()`.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1193)
 - Convert pred data to uint8 prior to imwrite png.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1185)
 - Minor correction to unsupported file extension error message.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1177)

## v2.9.6 (2022-06-02)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.5...release)

**Installation**

 - unify installation (attempt #2).  [View pull request](https://github.com/ivadomed/ivadomed/pull/1129)

**DOCUMENTATION**

 - Update documentation and config files to add WandB details.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1120)

**ENHANCEMENT**

 - Support for WandB Experimental Tracking .  [View pull request](https://github.com/ivadomed/ivadomed/pull/1069)

**CONTINUOUS INTEGRATION**
 
 - Adds various CI and testing improvement not impacting end users. 

## v2.9.5 (2022-04-06)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.4...release)

**BUG**

- Fix TSV metadata indexation and remove unused lines from bids_dataframe based on split_method.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1112)
- Fix loading of TIF 16bits grayscale files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1107)
- Fix loading when names of multiple target_suffix overlap.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1100)

**ENHANCEMENT**

- Add type hintings to fields inside all keywords(KW) dataclasses.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1109)

**DOCUMENTATION**

- Clarify data and loading documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1103)


## v2.9.4 (2022-03-09)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.3...release)

**FEATURE**

- Segment with ONNX or PT model based on CPU/GPU availability.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1086)

**ENHANCEMENT**

- Update microscopy following BEP release.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1025)

**BUG**

- Fixing mix-up for GPU training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1063)

- **REFACTORING**

- Refactor missing print statements to be using logger.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1085)
- Convert print to logger format for much more granular unified control..  [View pull request](https://github.com/ivadomed/ivadomed/pull/1040)
- Update pybids to 0.14.0.  [View pull request](https://github.com/ivadomed/ivadomed/pull/994)

**DOCUMENTATION**

- Add ADS use case in documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1080)
- Updated documentation for SoftSeg training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1064)
- Rewrite tutorial 2 with sphinx tab.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1045)
- Format revamped Tutorial 1 to highlight the CLI vs JSON approaches.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1039)
- Improve Installation Doc Readability based for Step 3 relating to GPU setup.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1037)

## v2.9.3 (2022-02-01)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.2...release)

**FEATURE**

- Apply filter parameters on 2D patches to remove empty patches.  [View pull request](https://github.com/ivadomed/ivadomed/pull/980)

**REFACTORING**

- Update pred_to_png prediction filenames for ADS integration.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1050)

**DOCUMENTATION**

- Instruction to update `"bids_config"` key in microscopy tutorial.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1053)


## v2.9.2 (2022-01-18)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.1...release)

**FEATURE**

- Implementation of Random Blur Augmentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1034)
- Implementation of Random Bias Field Augmentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1033)
- Implementation of Random Gamma Contrast Augmentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1015)

**DEPENDENCIES**

- Unpin `tensorboard` to avoid conflict with downstream SCT requirements.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1048)

**BUG**

- Rename prediction filenames: add class index and compat. for multi-rater.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1043)
- Fix pixel size keyword in run_segment_command.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1024)
- Replaced flip_axes with the correct bool element at index..  [View pull request](https://github.com/ivadomed/ivadomed/pull/1013)

**DOCUMENTATION**

- Add microscopy tutorial.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1036)
- Removed one child-headings for clarity.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1028)
- Typo fix for URL that is bricking the Colab link.  [View pull request](https://github.com/ivadomed/ivadomed/pull/1021)
- Experimental incorporation of tutorial jupyter notebooks open in Colab path.  [View pull request](https://github.com/ivadomed/ivadomed/pull/998)


## v2.9.1 (2021-12-13)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.9.0...release)

**ENHANCEMENT**

- Add forced indexation of "micr" datatype.  [View pull request](https://github.com/ivadomed/ivadomed/pull/995)
- Apply transforms on 2D patches.  [View pull request](https://github.com/ivadomed/ivadomed/pull/982)

**DOCUMENTATION**

- Update Tutorial 1/2/3 and readme.md to fix minor display issues.  [View pull request](https://github.com/ivadomed/ivadomed/pull/992)
- Update installation instruction to fit recent CUDA11 and torch 1.8+ push.  [View pull request](https://github.com/ivadomed/ivadomed/pull/969)

**REFACTORING**

- Fully Remove HeMIS model, Adaptive and h5py/HDF5.  [View pull request](https://github.com/ivadomed/ivadomed/pull/984)
- Use keywords for the rest of the files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/946)


## v2.9.0 (2021-11-14)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.8.0...release)

**ENHANCEMENT**

- Make ivadomed be compatible with python 3.9 and PyTorch 1.8.  [View pull request](https://github.com/ivadomed/ivadomed/pull/819)

**DEPENDENCIES**

- Pin to CUDA-11.  [View pull request](https://github.com/ivadomed/ivadomed/pull/951)

**BUG FIXES**

- Pin PyParsing version to be compatible with pip 20.  [View pull request](https://github.com/ivadomed/ivadomed/pull/987)
- Fix pytest test_download_data_no_dataset_specified fail bug.  [View pull request](https://github.com/ivadomed/ivadomed/pull/968)
- Fix GeneralizedDiceLoss with `include_background=true` and `batch_size>1` .  [View pull request](https://github.com/ivadomed/ivadomed/pull/962)
- Fix undo_transforms in volume reconstruction.  [View pull request](https://github.com/ivadomed/ivadomed/pull/957)
- Fix undo_transforms in image reconstruction.  [View pull request](https://github.com/ivadomed/ivadomed/pull/956)
- add metadata to create_metadata_dict.  [View pull request](https://github.com/ivadomed/ivadomed/pull/954)
- Update scripts in `dev/prepare_data` to use new SCT config syntax (`.yml`).  [View pull request](https://github.com/ivadomed/ivadomed/pull/949)
- Fix config loading errors.  [View pull request](https://github.com/ivadomed/ivadomed/pull/944)
- Fix dropout_rate key in models.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/937)
- Add additional check for incorrect final_activation value.  [View pull request](https://github.com/ivadomed/ivadomed/pull/933)
- Make ivadomed be compatible with python3.9 and PyTorch 1.8.  [View pull request](https://github.com/ivadomed/ivadomed/pull/819)

**DOCUMENTATION**

- Minor modifications to the documentation for tutorial 3.  [View pull request](https://github.com/ivadomed/ivadomed/pull/988)
- Fix resample axis order in documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/978)
- Update help.rst.  [View pull request](https://github.com/ivadomed/ivadomed/pull/967)
- Fixing issues in estimate uncertainty tutorial.  [View pull request](https://github.com/ivadomed/ivadomed/pull/936)
- Fix link to data file in ivadomed instructions.  [View pull request](https://github.com/ivadomed/ivadomed/pull/929)
- Fixes object detection path in cascaded architecture tutorial.  [View pull request](https://github.com/ivadomed/ivadomed/pull/922)
- Make ivadomed be compatible with python3.9 and PyTorch 1.8.  [View pull request](https://github.com/ivadomed/ivadomed/pull/819)

**REFACTORING**

- Fully Remove HeMIS model, Adaptive and h5py/HDF5.  [View pull request](https://github.com/ivadomed/ivadomed/pull/984)
- Fix path_output in automated training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/914)
- Using keywords for ivadomed/scripts folder.  [View pull request](https://github.com/ivadomed/ivadomed/pull/934)
- Keywords refactoring Phase II: loader focus.  [View pull request](https://github.com/ivadomed/ivadomed/pull/909)
- Adopting pathllib for loader/bids_dataframe.  [View pull request](https://github.com/ivadomed/ivadomed/pull/947)
- Adopting pathlib for tests.  [View pull request](https://github.com/ivadomed/ivadomed/pull/901)
- Adopting pathlib training.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/897)
- Adopting pathlib for main.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/892)
- Adopting pathlib for loader/utils.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/879)

**TESTING**

- Fix pytest test_download_data_no_dataset_specified fail bug.  [View pull request](https://github.com/ivadomed/ivadomed/pull/968)

**CI**

- Update Sphinx dependency version and check RTD.org performance.  [View pull request](https://github.com/ivadomed/ivadomed/pull/974)
- Fix pytest problem.  [View pull request](https://github.com/ivadomed/ivadomed/pull/968)
- Update to GitHub Action to use `setup-python@v2`.  [View pull request](https://github.com/ivadomed/ivadomed/pull/959)
- Make ivadomed be compatible with python3.9 and PyTorch 1.8.  [View pull request](https://github.com/ivadomed/ivadomed/pull/819)

## v2.8.0 (2021-08-31)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.7.4...v.2.8.0)

**FEATURE**

- Add image reconstruction from 2D patches.  [View pull request](https://github.com/ivadomed/ivadomed/pull/782)
- Add sha256 for training data.  [View pull request](https://github.com/ivadomed/ivadomed/pull/760)

**CI**

- Exclude testing directory in coveralls.  [View pull request](https://github.com/ivadomed/ivadomed/pull/776)
- Improve current GitHub Action CI with multi OS support.  [View pull request](https://github.com/ivadomed/ivadomed/pull/757)

**BUG**

- Fix training_curve.py output.  [View pull request](https://github.com/ivadomed/ivadomed/pull/923)
- Fix inverted dimensions in microscopy pixelsize.  [View pull request](https://github.com/ivadomed/ivadomed/pull/916)
- Fix segment functions for models without pre-processing transforms.  [View pull request](https://github.com/ivadomed/ivadomed/pull/874)
- Fix microscopy ground-truth range of values.  [View pull request](https://github.com/ivadomed/ivadomed/pull/870)
- `utils.py`: Only raise ArgParseException for non-zero SystemExits.  [View pull request](https://github.com/ivadomed/ivadomed/pull/854)
- Remove `anaconda` from explicit dependencies.  [View pull request](https://github.com/ivadomed/ivadomed/pull/845)
- Fix multiclass evaluation bug.  [View pull request](https://github.com/ivadomed/ivadomed/pull/837)
- Fix last slice missing in testing bug.  [View pull request](https://github.com/ivadomed/ivadomed/pull/835)
- Skip all NumpyToTensor transformation for retrocompatibility.  [View pull request](https://github.com/ivadomed/ivadomed/pull/830)
- Remove all NumpyToTensor configs keys.  [View pull request](https://github.com/ivadomed/ivadomed/pull/826)
- Add missing "-r" flags to installation.rst.  [View pull request](https://github.com/ivadomed/ivadomed/pull/820)
- Call NumpyToTensor last.  [View pull request](https://github.com/ivadomed/ivadomed/pull/818)
- Fix bug in loader for multiple raters.  [View pull request](https://github.com/ivadomed/ivadomed/pull/806)
- Hot patch to address Inference issue #803.  [View pull request](https://github.com/ivadomed/ivadomed/pull/804)
- Add tmp and log file to gitignore.  [View pull request](https://github.com/ivadomed/ivadomed/pull/794)

**INSTALLATION**

- Remove `anaconda` from explicit dependencies.  [View pull request](https://github.com/ivadomed/ivadomed/pull/845)

**DOCUMENTATION**

- Fix neuropoly guidelines link in ivadomed contribution guidelines document.  [View pull request](https://github.com/ivadomed/ivadomed/pull/924)
- Change readme to point to the latest build version.  [View pull request](https://github.com/ivadomed/ivadomed/pull/875)
- Installation instruction steps explicity recommended for MacOS but not Linux.  [View pull request](https://github.com/ivadomed/ivadomed/pull/847)
- Clarified step 2 for pytorch/torchvision.  [View pull request](https://github.com/ivadomed/ivadomed/pull/842)
- Add missing "-r" flags to installation.rst.  [View pull request](https://github.com/ivadomed/ivadomed/pull/820)
- Update one class segmentation tutorial's output and segmentation image.  [View pull request](https://github.com/ivadomed/ivadomed/pull/779)
- Update documentation with the solution to failing test_adaptive.py on MacOS.  [View pull request](https://github.com/ivadomed/ivadomed/pull/771)
- Added link to JOSS paper.  [View pull request](https://github.com/ivadomed/ivadomed/pull/748)

**DEPENDENCIES**

- Remove `anaconda` from explicit dependencies.  [View pull request](https://github.com/ivadomed/ivadomed/pull/845)

**ENHANCEMENT**

- Fix training_curve.py output.  [View pull request](https://github.com/ivadomed/ivadomed/pull/923)
- Fix microscopy ground-truth range of values.  [View pull request](https://github.com/ivadomed/ivadomed/pull/870)
- Fix generate_sha_256 for joblib files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/866)
- Add microscopy config file.  [View pull request](https://github.com/ivadomed/ivadomed/pull/850)
- Add the inference steps for PNG/TIF microscopy data.  [View pull request](https://github.com/ivadomed/ivadomed/pull/834)
- New loader: Load PNG/TIF/JPG microscopy files as Nibabel objects.  [View pull request](https://github.com/ivadomed/ivadomed/pull/813)
- Speed up IvadoMed Import Speed.  [View pull request](https://github.com/ivadomed/ivadomed/pull/793)
- Remove data dependencies from `if` statements in the `Decoder()` forward pass.  [View pull request](https://github.com/ivadomed/ivadomed/pull/752)

**TESTING**

- Unsilence test_rbg.  [View pull request](https://github.com/ivadomed/ivadomed/pull/832)
- Fix test_sampler.  [View pull request](https://github.com/ivadomed/ivadomed/pull/831)
- Fix bug in loader for multiple raters.  [View pull request](https://github.com/ivadomed/ivadomed/pull/806)
- Exclude testing directory in coveralls.  [View pull request](https://github.com/ivadomed/ivadomed/pull/776)
- Update documentation with the solution to failing test_adaptive.py on MacOS.  [View pull request](https://github.com/ivadomed/ivadomed/pull/771)
- Migrate test_segment_volume.py from unit_tests to functional_tests.  [View pull request](https://github.com/ivadomed/ivadomed/pull/767)
- Improve current GitHub Action CI with multi OS support.  [View pull request](https://github.com/ivadomed/ivadomed/pull/757)

**REFACTORING**

- Extract class SliceFilter, BalancedSample and SampleMetaData from loader.util.  [View pull request](https://github.com/ivadomed/ivadomed/pull/928)
- Extracted BidsDataFrame class outside of loader/utils.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/917)
- Initialize the adoption of centralized management of keywords via keywords.py (Phase I: compilation of all keywords).  [View pull request](https://github.com/ivadomed/ivadomed/pull/904)
- Fix empty list default parameter antipattern..  [View pull request](https://github.com/ivadomed/ivadomed/pull/903)
- Pathlib adoption for visualize.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/900)
- Pathlib adoption for utils.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/899)
- Pathlib adoption for uncertainty.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/898)
- Pathlib adoption for testing.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/896)
- Pathlib adoption for postprocessing.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/895)
- Pathlib adoption for models.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/894)
- Pathlib adoption for mixup.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/893)
- Pathlib adoption for inference.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/891)
- Pathlib adoption for evaluation.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/890)
- pathlib config_manager.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/889)
- Pathlib adoption for visualize_transform.  [View pull request](https://github.com/ivadomed/ivadomed/pull/888)
- Pathlib adoption for visualize_and_compare_testing_models.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/887)
- Pathlib adoption for script/training_curve.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/886)
- Pathlib adoption for script/prepare_dataset_vertibral_labeling.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/885)
- Pathlib adoption for extract_small_dataset.  [View pull request](https://github.com/ivadomed/ivadomed/pull/884)
- pathlib for download_data.  [View pull request](https://github.com/ivadomed/ivadomed/pull/883)
- pathlib for script/automate_training.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/881)
- pathlib change for object_detection/utils.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/880)
- Pathlib adoption for loader/segmentation_pair.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/878)
- Pathlib adoption for loader/film.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/877)
- Pathlib adoption for adaptative.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/876)
- Update config_bids.json following changes in microscopy BEP.  [View pull request](https://github.com/ivadomed/ivadomed/pull/838)
- Extracted Loader Classes into separate files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/828)
- Refactor segment_volume to reduce complexity.  [View pull request](https://github.com/ivadomed/ivadomed/pull/791)
- Refactoring: BidsDataset __init__ reduce complexity .  [View pull request](https://github.com/ivadomed/ivadomed/pull/765)
- Refactoring: reduce complexity of BIDStoHDF5 _load_filenames.  [View pull request](https://github.com/ivadomed/ivadomed/pull/737)

## v2.7.4 (2021-03-15)

See `2.7.3`. We had to re-release because the GitHub Action didn't get triggered to push the release
to `PyPI` as it started as a draft. See here for more details:

[GitHub Actions Bug](https://github.community/t/workflow-set-for-on-release-not-triggering-not-showing-up/16286)


## v2.7.3 (2021-03-15)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.7.2...release)

**BUG**

 - Copy nibabel header when creating output prediction.  [View pull request](https://github.com/ivadomed/ivadomed/pull/714)
 - Dynamically write dataset_description.json file to suppress pybids warning.  [View pull request](https://github.com/ivadomed/ivadomed/pull/690)

**DOCUMENTATION**

 - Change archive links to repository links for pre-trained models.  [View pull request](https://github.com/ivadomed/ivadomed/pull/700)

**ENHANCEMENT**

 - New loader: Refactor BidsDataset classes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/691)


## v2.7.2 (2021-02-19)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.7.1...v2.7.2)

**BUG**

 - Multiclass ignored during inference if n_input and n_output are different.  [View pull request](https://github.com/ivadomed/ivadomed/pull/688)
 - Merged participants.tsv file saving bug correction.  [View pull request](https://github.com/ivadomed/ivadomed/pull/684)
 - Make change_keys method from ConfigurationManager compatible with python3.8.  [View pull request](https://github.com/ivadomed/ivadomed/pull/681)

**DOCUMENTATION**

 - Add DOI JOSS.  [View pull request](https://github.com/ivadomed/ivadomed/pull/683)
 - Adding Zenodo DOI.  [View pull request](https://github.com/ivadomed/ivadomed/pull/677)

**ENHANCEMENT**

 - New loader: input from multiple BIDS datasets.  [View pull request](https://github.com/ivadomed/ivadomed/pull/687)
 - Add pre-commit hooks to limit file size to 500KB .  [View pull request](https://github.com/ivadomed/ivadomed/pull/682)
 - Shared weights for the two first FiLM generator layers.  [View pull request](https://github.com/ivadomed/ivadomed/pull/679)
 - Allow for non-dictionary hyperparameters in automate_training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/661)

**FEATURE**

 - Enable the pipeline to run with inputs from multiple BIDS datasets.  [View pull request](https://github.com/ivadomed/ivadomed/pull/588)

## v2.7.1 (2021-02-09)
[View change](https://github.com/ivadomed/ivadomed/pull/676)

## v2.7.0 (2021-02-09)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.6.1...v2.7.0)

**BUG**

 - Fix structure wise uncertainty computation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/664)
 - Fix bugs in plot film params.  [View pull request](https://github.com/ivadomed/ivadomed/pull/646)
 - Change condition to save FiLM parameters .  [View pull request](https://github.com/ivadomed/ivadomed/pull/645)
 - Fix store film params.  [View pull request](https://github.com/ivadomed/ivadomed/pull/642)
 - soft_gt param: only active after Data Augmentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/624)
 - AnimatedGIf import and documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/623)
 - Fix pandas typecast issue in test_split_dataset.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/606)
 - Make sure test_HeMIS runs tests in order.  [View pull request](https://github.com/ivadomed/ivadomed/pull/602)
 - Fix loader/adaptative.py code with reading/writing HDF5 files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/592)
 - Automate_training: fix bug for multiple parameters.  [View pull request](https://github.com/ivadomed/ivadomed/pull/586)
 - Load 2D GT slice as uint if not soft training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/582)

**DOCUMENTATION**

 - Updated affiliations, Added Marie-Helene.  [View pull request](https://github.com/ivadomed/ivadomed/pull/674)
 - Fix missing dilate-gt.png.  [View pull request](https://github.com/ivadomed/ivadomed/pull/653)
 - Reformat configuration_file.rst for docs.  [View pull request](https://github.com/ivadomed/ivadomed/pull/650)
 - Add metavar to parser.  [View pull request](https://github.com/ivadomed/ivadomed/pull/641)
 - Add a README for the Sphinx docs.  [View pull request](https://github.com/ivadomed/ivadomed/pull/626)
 - Add documentation on packaged model format.  [View pull request](https://github.com/ivadomed/ivadomed/pull/625)
 - Add the Twitter badge.  [View pull request](https://github.com/ivadomed/ivadomed/pull/622)
 - Add new custom css rule for table in purpose section (#617).  [View pull request](https://github.com/ivadomed/ivadomed/pull/619)
 - Add DeepReg to comparison table.  [View pull request](https://github.com/ivadomed/ivadomed/pull/618)
 - Update PyTorch Ref.  [View pull request](https://github.com/ivadomed/ivadomed/pull/616)
 - Small clarifications and typos fixes in the Unet tutorial.  [View pull request](https://github.com/ivadomed/ivadomed/pull/610)
 - Added warning on installation to make sure proper Python version is installed.  [View pull request](https://github.com/ivadomed/ivadomed/pull/607)
 - Made the BIDS example more general for the audience.  [View pull request](https://github.com/ivadomed/ivadomed/pull/597)

**ENHANCEMENT**

 - Add new keys config manager.  [View pull request](https://github.com/ivadomed/ivadomed/pull/668)
 - Store FiLM parameters during testing instead of training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/663)
 - Externalize command, log_directory, and bids_path fields from JSON config files to CLI.  [View pull request](https://github.com/ivadomed/ivadomed/pull/652)
 - New loader: BidsDataframe class.  [View pull request](https://github.com/ivadomed/ivadomed/pull/648)
 - version_info.log  added in the log directory.  [View pull request](https://github.com/ivadomed/ivadomed/pull/639)
 - Indicate folder created after running ivadomed_download_data.  [View pull request](https://github.com/ivadomed/ivadomed/pull/609)
 - Add explanation for Windows incompatibility in installation docs.  [View pull request](https://github.com/ivadomed/ivadomed/pull/605)
 - Specify Python version in setup.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/603)
 - Add new filter to SliceFilter class.  [View pull request](https://github.com/ivadomed/ivadomed/pull/594)
 - New loader: Adapt splitting methods.  [View pull request](https://github.com/ivadomed/ivadomed/pull/591)

**TESTING**

 - Add functional test for automate_training run_test flag.  [View pull request](https://github.com/ivadomed/ivadomed/pull/647)
 - Add test template files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/638)
 - Remove the testing_data folder from ivadomed.  [View pull request](https://github.com/ivadomed/ivadomed/pull/631)
 - Bug in Coveralls release 3.0.0.  [View pull request](https://github.com/ivadomed/ivadomed/pull/628)
 - Add tests for create_bids_dataframe function.  [View pull request](https://github.com/ivadomed/ivadomed/pull/584)

**REFACTORING**

 - Reformat configuration_file.rst for docs.  [View pull request](https://github.com/ivadomed/ivadomed/pull/650)
 - New loader: BidsDataframe class.  [View pull request](https://github.com/ivadomed/ivadomed/pull/648)
 - Standardize the gpu ID argument.  [View pull request](https://github.com/ivadomed/ivadomed/pull/644)
 - Unit Test cleanup.  [View pull request](https://github.com/ivadomed/ivadomed/pull/636)
 - Remove test_script and ivado_functional_test files.  [View pull request](https://github.com/ivadomed/ivadomed/pull/634)

## v2.6.1 (2020-12-15)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.6.0...v2.6.1)

**BUG**

 - Fix missing attribute softmax.  [View pull request](https://github.com/ivadomed/ivadomed/pull/547)
 - Split_dataset: consider center_list when per_patient is used.  [View pull request](https://github.com/ivadomed/ivadomed/pull/537)

**DOCUMENTATION**

 - Make usage clearer.  [View pull request](https://github.com/ivadomed/ivadomed/pull/578)
 - Removing support for Python 3.9 (for now).  [View pull request](https://github.com/ivadomed/ivadomed/pull/562)
 - Updating comparison table after review.  [View pull request](https://github.com/ivadomed/ivadomed/pull/560)

**ENHANCEMENT**

 - Remove small for multiclass.  [View pull request](https://github.com/ivadomed/ivadomed/pull/570)
 - Save config file before training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/569)
 - Apply bounding box safety factor in segment volume.  [View pull request](https://github.com/ivadomed/ivadomed/pull/549)
 - Multichannel support for convert_to_onnx script.  [View pull request](https://github.com/ivadomed/ivadomed/pull/544)

**FEATURE**

 - Select subjects for training based on metadata.  [View pull request](https://github.com/ivadomed/ivadomed/pull/534)

## v2.6.0 (2020-11-23)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.5.0...v2.6.0)

**BUG**

 - Make is_2d retrocompatibility.  [View pull request](https://github.com/ivadomed/ivadomed/pull/535)
 - Support multiclass if first class missing.  [View pull request](https://github.com/ivadomed/ivadomed/pull/522)

**DOCUMENTATION**

 - AdapWing 3D: fix comment.  [View pull request](https://github.com/ivadomed/ivadomed/pull/531)
 - paper.md: overview_title.png path.  [View pull request](https://github.com/ivadomed/ivadomed/pull/529)
 - paper.bib: correct typo.  [View pull request](https://github.com/ivadomed/ivadomed/pull/528)
 - Fix DOIs in paper.bib.  [View pull request](https://github.com/ivadomed/ivadomed/pull/527)
 - Redirect to DokuWiki/GitHub from the contributing guidelines.  [View pull request](https://github.com/ivadomed/ivadomed/pull/523)
 - Change path for images.  [View pull request](https://github.com/ivadomed/ivadomed/pull/521)

**ENHANCEMENT**

 - automate_training: add new parameter to change multiple params.  [View pull request](https://github.com/ivadomed/ivadomed/pull/533)
 - Softseg multiclass.  [View pull request](https://github.com/ivadomed/ivadomed/pull/530)
 - Multiclass and multichannel support for segment volume.  [View pull request](https://github.com/ivadomed/ivadomed/pull/524)

**FEATURE**

 - Create sample to balance metadata.  [View pull request](https://github.com/ivadomed/ivadomed/pull/503)

## v2.5.0 (2020-11-10)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.4.0...v2.5.0)

**BUG**

 - paper.md: Fixed broken link.  [View pull request](https://github.com/ivadomed/ivadomed/pull/517)
 - Change default value of config.json.  [View pull request](https://github.com/ivadomed/ivadomed/pull/514)

**DEPENDENCIES**

 - Requirements.txt: force onnxruntime version.  [View pull request](https://github.com/ivadomed/ivadomed/pull/505)
 - set h5py version in requirements.txt.  [View pull request](https://github.com/ivadomed/ivadomed/pull/500)

**DOCUMENTATION**

 - JOSS submission.  [View pull request](https://github.com/ivadomed/ivadomed/pull/502)

**ENHANCEMENT**

 - Some fixes to logging.  [View pull request](https://github.com/ivadomed/ivadomed/pull/509)

**FEATURE**

 - Training without test set.  [View pull request](https://github.com/ivadomed/ivadomed/pull/498)
 - FiLM for 3D Unet.  [View pull request](https://github.com/ivadomed/ivadomed/pull/491)

**REFACTORING**

 - Refactor utils.py.  [View pull request](https://github.com/ivadomed/ivadomed/pull/497)

## v2.4.0 (2020-10-27)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.3.1...v2.4.0)

**BUG**

 - Fix missing version.txt in wheels package.  [View pull request](https://github.com/ivadomed/ivadomed/pull/488)

**DOCUMENTATION**

 - Added reference to arXiv citation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/485)
 - Documenting release workflow.  [View pull request](https://github.com/ivadomed/ivadomed/pull/483)

**ENHANCEMENT**

 - Option to override postprocessing in segment volume.  [View pull request](https://github.com/ivadomed/ivadomed/pull/486)
 - Configuration File Manager.  [View pull request](https://github.com/ivadomed/ivadomed/pull/484)


## v2.3.1 (2020-10-19)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.3.0...v2.3.1)

**BUG**

 - Version format.  [View pull request](https://github.com/ivadomed/ivadomed/pull/481)

## v2.3.0 (2020-10-19)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.2.1...v2.3.0)

**BUG**

 - Adapt all metrics to multiclass predictions.  [View pull request](https://github.com/ivadomed/ivadomed/pull/472)
 - fix run_test gpu assignation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/453)

**DOCUMENTATION**

 - Improving documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/477)
 - Tutorial fix.  [View pull request](https://github.com/ivadomed/ivadomed/pull/461)

**ENHANCEMENT**

 - Download data: Add models.  [View pull request](https://github.com/ivadomed/ivadomed/pull/476)
 - Refactoring: Changing print and exit to raise error.  [View pull request](https://github.com/ivadomed/ivadomed/pull/467)
 - Remove "eval" cmd.  [View pull request](https://github.com/ivadomed/ivadomed/pull/465)
 - Custom final activation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/458)
 - Display version.  [View pull request](https://github.com/ivadomed/ivadomed/pull/456)

**FEATURE**

 - Use custom data for film.  [View pull request](https://github.com/ivadomed/ivadomed/pull/460)
 - Uncertainty as post-processing step.  [View pull request](https://github.com/ivadomed/ivadomed/pull/459)

## v2.2.1 (2020-09-22)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.2.0...v2.2.1)

**BUG**

 - Cover image path change on README.  [View pull request](https://github.com/ivadomed/ivadomed/pull/451)

## v2.2.0 (2020-09-22)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.1.0...v2.2.0)

**BUG**

 - Minor fixes prior release.  [View pull request](https://github.com/ivadomed/ivadomed/pull/449)

**DEPENDENCIES**

 - Modify scripts/training_curve.py to avoid tensorflow dependency.  [View pull request](https://github.com/ivadomed/ivadomed/pull/396)

**DOCUMENTATION**

 - Updating documentation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/425)
 - Tutorial on uncertainty estimation.  [View pull request](https://github.com/ivadomed/ivadomed/pull/399)
 - Tutorial cascaded architecture.  [View pull request](https://github.com/ivadomed/ivadomed/pull/389)

**ENHANCEMENT**

 - Retrain model without resetting weights.  [View pull request](https://github.com/ivadomed/ivadomed/pull/447)
 - Normalized ReLU.  [View pull request](https://github.com/ivadomed/ivadomed/pull/384)
 - Create Ivadomed download function.  [View pull request](https://github.com/ivadomed/ivadomed/pull/379)

**FEATURE**

 - Evenly distribute subjects according to metadata.  [View pull request](https://github.com/ivadomed/ivadomed/pull/423)
 - Resume training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/416)
 - Find optimal threshold with ROC analysis.  [View pull request](https://github.com/ivadomed/ivadomed/pull/383)
 - Generate GIF during training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/374)
 - Add classifier model .  [View pull request](https://github.com/ivadomed/ivadomed/pull/278)

**TESTING**

 - Create coverage and improve testing.  [View pull request](https://github.com/ivadomed/ivadomed/pull/385)

## v2.1.0 (2020-07-21)
[View detailed changelog](https://github.com/ivadomed/ivadomed/compare/v2.0.2...v2.1.0)

**BUG**

 - Automate training seed.  [View pull request](https://github.com/ivadomed/ivadomed/pull/366)
 - Automate training bug.  [View pull request](https://github.com/ivadomed/ivadomed/pull/363)
 - Apply preprocessing after filter ROI.  [View pull request](https://github.com/ivadomed/ivadomed/pull/342)
 - Fix bug in automate training.  [View pull request](https://github.com/ivadomed/ivadomed/pull/339)
 - Transformations at test time: minor fixes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/335)

**DOCUMENTATION**

 - Documentation: metric more formal defintion.  [View pull request](https://github.com/ivadomed/ivadomed/pull/357)
 - Fix few documentation issues, add content.  [View pull request](https://github.com/ivadomed/ivadomed/pull/341)
 - Soft training: minor fixes.  [View pull request](https://github.com/ivadomed/ivadomed/pull/334)
 - Tutorial 01: One class segmentation 2D Unet.  [View pull request](https://github.com/ivadomed/ivadomed/pull/309)

**ENHANCEMENT**

 - Split dataset with no test center specified.  [View pull request](https://github.com/ivadomed/ivadomed/pull/370)
 - showing time after training (begin/end/duration).  [View pull request](https://github.com/ivadomed/ivadomed/pull/365)
 - Optimize binarization.  [View pull request](https://github.com/ivadomed/ivadomed/pull/364)
 - Automate training improvement.  [View pull request](https://github.com/ivadomed/ivadomed/pull/362)
 - Simplify code when filtering ROI.  [View pull request](https://github.com/ivadomed/ivadomed/pull/361)
 - Scripts: Add entry points, modify doc display, and started to add github action testing.  [View pull request](https://github.com/ivadomed/ivadomed/pull/328)
