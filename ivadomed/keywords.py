from dataclasses import dataclass


@dataclass(frozen=True)
class ConfigKW:
    LOADER_PARAMETERS = "loader_parameters"
    TRAINING_PARAMETERS = "training_parameters"
    MODEL_NAME = "model_name"
    MODIFIED_3D_UNET = "Modified3DUNet"
    DEBUGGING = "debugging"
    WANDB = "wandb"
    FILMED_UNET = "FiLMedUnet"
    DEFAULT_MODEL = "default_model"
    OBJECT_DETECTION_PARAMS = "object_detection_params"
    GPU_IDS = "gpu_ids"
    PATH_OUTPUT = "path_output"
    POSTPROCESSING = "postprocessing"
    COMMAND = "command"
    LOG_FILE = "log_file"
    TRANSFORMATION = "transformation"
    SPLIT_DATASET = "split_dataset"
    UNCERTAINTY = "uncertainty"
    UNDO_TRANSFORMS = "undo_transforms"
    EVALUATION_PARAMETERS = "evaluation_parameters"
    HEMIS_UNET = "HeMISUnet"
    SPLIT_PATH = "split_path"
    TRAINING_SHA256 = "training_sha256"


@dataclass(frozen=True)
class WandbKW:
    WANDB_API_KEY = "wandb_api_key"
    PROJECT_NAME = "project_name"
    GROUP_NAME = "group_name"
    RUN_NAME = "run_name"
    LOG_GRADS_EVERY = "log_grads_every"


@dataclass(frozen=True)
class LoaderParamsKW:
    PATH_DATA: str = "path_data"
    BIDS_CONFIG: str = "bids_config"
    TARGET_SUFFIX: str = "target_suffix"
    ROI_PARAMS: str = "roi_params"
    CONTRAST_PARAMS: str = "contrast_params"
    MULTICHANNEL: str = "multichannel"  # boolean key that is used to change the configuration file ever slightly.
    EXTENSIONS: str = "extensions"
    TARGET_GROUND_TRUTH: str = "target_ground_truth"
    TARGET_SESSIONS: str = "target_sessions"  # the sessions to focus the analyses on
    METADATA_TYPE: str = "metadata_type"
    MODEL_PARAMS: str = "model_params"
    SLICE_AXIS: str = "slice_axis"
    IS_INPUT_DROPOUT: str = "is_input_dropout"
    SLICE_FILTER_PARAMS: str = "slice_filter_params"
    SUBJECT_SELECTION: str = "subject_selection"
    MULTI_PATH_LOADER: str = "multi_loader"
    TRADITIONAL_BIDS_LOADER: str = "traditional_bids_loader"


@dataclass(frozen=True)
class SplitDatasetKW:
    SPLIT_METHOD: str = "split_method"
    FNAME_SPLIT: str = "fname_split"
    DATA_TESTING: str = "data_testing"
    RANDOM_SEED: str = "random_seed"
    TRAIN_FRACTION: str = "train_fraction"
    TEST_FRACTION: str = "test_fraction"
    BALANCE: str = "balance"


@dataclass(frozen=True)
class DataTestingKW:
    DATA_TYPE: str = "data_type"
    DATA_VALUE: str = "data_value"


@dataclass(frozen=True)
class TrainingParamsKW:
    BALANCE_SAMPLES: str = "balance_samples"
    BATCH_SIZE: str = "batch_size"


@dataclass(frozen=True)
class TransformationKW:
    ROICROP: str = "ROICrop"
    CENTERCROP: str = "CenterCrop"
    RESAMPLE: str = "Resample"
    RANDOM_AFFINE: str = "RandomAffine"


@dataclass(frozen=True)
class BalanceSamplesKW:
    APPLIED: str = "applied"
    TYPE: str = "type"


@dataclass(frozen=True)
class ContrastParamsKW:
    CONTRAST_LST: str = "contrast_lst"  # The list help determine the number of model parameter inputs.
    BALANCE: str = "balance"
    TRAINING_VALIDATION: str = "training_validation"
    TESTING: str = "testing"

@dataclass(frozen=True)
class ModelParamsKW:
    LENGTH_2D: str = "length_2D"
    STRIDE_2D: str = "stride_2D"
    LENGTH_3D: str = "length_3D"
    STRIDE_3D: str = "stride_3D"
    FILM_LAYERS: str = "film_layers"
    FOLDER_NAME: str = "folder_name"
    METADATA: str = "metadata"
    FILM_ONEHOTENCODER: str = "film_onehotencoder"
    N_METADATA: str = "n_metadata"
    APPLIED: str = "applied"
    NAME: str = "name"
    IS_2D: str = "is_2d"
    IN_CHANNEL: str = "in_channel"
    OUT_CHANNEL: str = "out_channel"
    TARGET_LST: str = "target_lst"
    ROI_LST: str = "roi_lst"
    PATH_HDF5: str = "path_hdf5"
    CSV_PATH: str = "csv_path"
    RAM: str = "ram"
    ATTENTION: str = "attention"
    DEPTH: str = "depth"
    MISSING_PROBABILITY: str = "missing_probability"
    MISSING_PROBABILITY_GROWTH: str = "missing_probability_growth"
    DROPOUT_RATE: str = "dropout_rate"
    N_FILTERS: str = "n_filters"

@dataclass(frozen=True)
class SubjectDictKW:
    ABSOLUTE_PATHS: str = "absolute_paths"
    DERIV_PATH: str = "deriv_path"
    ROI_FILENAME: str = "roi_filename"
    METADATA: str = "metadata"
    EXTENSIONS: str = "extensions"


@dataclass(frozen=True)
class SubjectDataFrameKW:
    FILENAME: str = "filename"


@dataclass(frozen=True)
class OptionKW:
    METADATA: str = "metadata"
    FNAME_PRIOR: str = 'fname_prior'
    BINARIZE_PREDICTION: str = "binarize_prediction"
    BINARIZE_MAXPOOLING: str = "binarize_maxpooling"
    KEEP_LARGEST: str = "keep_largest"
    FILL_HOLES: str = "fill_holes"
    REMOVE_SMALL: str = "remove_small"
    OVERLAP_2D: str = "overlap_2D"
    PIXEL_SIZE: str = "pixel_size"
    PIXEL_SIZE_UNITS: str = "pixel_size_units"
    NO_PATCH: str = "no_patch"


@dataclass(frozen=True)
class BidsDataFrameKW:
    # bids layout converted to dataframe during bids dataset creation
    PATH: str = "path"   # full path.
    FILENAME: str = "filename"  # the actual file's name (base)
    PARTICIPANT_I: str = "participant_id"  # i.e.    sub-unf01
    SUBJECT: str = "subject"  # i.e.  unf01
    SUFFIX: str = "suffix"   # T1w
    SESSION: str = "session"  # session field (single int) in Bids DataFrame
    EXTENSION: str = "extension"   # .nii.gz
    DERIVATIVES: str = "derivatives"


@dataclass(frozen=True)
class ROIParamsKW:
    SUFFIX: str = "suffix"
    SLICE_FILTER_ROI: str = "slice_filter_roi"


@dataclass(frozen=True)
class MetadataKW:
    CONTRAST: str = "contrast"
    CONTRASTS: str = "contrasts"
    BOUNDING_BOX: str = "bounding_box"
    DATA_TYPE: str = "data_type"
    PRE_RESAMPLE_SHAPE: str = "preresample_shape"
    CROP_PARAMS: str = "crop_params"
    MRI_PARAMS: str = "mri_params"
    ROTATION: str = "rotation"
    TRANSLATION: str = "translation"
    SCALE: str = "scale"
    COORD: str = "coord"
    ZOOMS: str = "zooms"
    UNDO: str = "undo"
    REVERSE: str = "reverse"
    OFFSET: str = "offset"
    ELASTIC: str = "elastic"
    GAUSSIAN_NOISE: str = "gaussian_noise"
    GAMMA: str = "gamma"
    BIAS_FIELD: str = "bias_field"
    BLUR: str = "blur"
    DATA_SHAPE: str = "data_shape"
    SLICE_INDEX: str = "slice_index"
    MISSING_MOD: str = "missing_mod"
    METADATA_DICT: str = "metadata_dict"
    INDEX_SHAPE: str = "index_shape"
    GT_METADATA: str = "gt_metadata"
    GT_FILENAMES: str = "gt_filenames"
    INPUT_METADATA: str = "input_metadata"
    INPUT_FILENAMES: str = "input_filenames"
    ROI_METADATA: str = "roi_metadata"
    PIXEL_SIZE: str = "PixelSize"
    PIXEL_SIZE_UNITS: str = "PixelSizeUnits"


@dataclass(frozen=True)
class ObjectDetectionParamsKW:
    GPU_IDS: str = "gpu_ids"
    PATH_OUTPUT: str = "path_output"
    OBJECT_DETECTION_PATH: str = "object_detection_path"
    SAFETY_FACTOR: str = "safety_factor"


@dataclass(frozen=True)
class UncertaintyKW:
    ALEATORIC: str = 'aleatoric'
    N_IT: str = "n_it"


@dataclass(frozen=True)
class PostprocessingKW:
    BINARIZE_PREDICTION: str = "binarize_prediction"


@dataclass(frozen=True)
class BinarizeProdictionKW:
    THR: str = "thr"


@dataclass(frozen=True)
class SliceFilterParamsKW:
    FILTER_EMPTY_MASK: str = "filter_empty_mask"


@dataclass(frozen=True)
class IgnoredFolderKW:
    MACOSX: str = "__MACOSX"


@dataclass(frozen=True)
class MetricsKW:
    RECALL_SPECIFICITY: str = "recall_specificity"
    DICE: str = "dice"

@dataclass(frozen=True)
class MetadataParamsKW:
    CONTRAST = "contrast"
    BOUNDING_BOX = "bounding_box"

@dataclass(frozen=True)
class MetadataChoiceKW:
    MRI_PARAMS = "mri_params"
    CONTRASTS = "contrasts"

@dataclass(frozen=True)
class SegmentationDatasetKW:
    X_MIN: str = 'x_min'
    X_MAX: str = 'x_max'
    Y_MIN: str = 'y_min'
    Y_MAX: str = 'y_max'
    Z_MIN: str = 'z_min'
    Z_MAX: str = 'z_max'
    HANDLER_INDEX: str = 'handler_index'

@dataclass(frozen=True)
class SegmentationPairKW:
    GT_METADATA = "gt_metadata"
    INPUT_METADATA = "input_metadata"
    ROI_METADATA = "roi_metadata"
    GT = "gt"
    INPUT = "input"
    ROI = "roi"

@dataclass(frozen=True)
class CommandKW:
    TRAIN = "train"
    SEGMENT = "segment"
    TEST = "test"

@dataclass(frozen=True)
class DataloaderKW:
    # Most of these keywords are used in V2 Dataloader to support cross BIDS/Regex/File Dataset loading.

    # AllDatasetGroups Level
    DATASET_GROUPS: str = "dataset_groups"

    # DatasetGroup Level
    DATASET_GROUP_LABEL: str = "dataset_group_label"
    TRAINING: str = "training"
    VALIDATION: str = "validation"
    TESTING: str = "testing"
    # Reserved DatasetGroup to be implemented
    TRAINING_VALIDATION: str = "training_validation"
    TRAINING_TESTING: str = "training_testing"
    TRAINING_VALIDATION_TESTING: str = "training_validation_testing"

    # Dataset Level
    TYPE: str = "type"
    DATASET_LABEL: str = "dataset_label"
    INPUT_GT: str = "input_gt"
    EXPECTED_INPUT: str = "expected_input"
    EXPECTED_GT: str = "expected_gt"
    # Reserved keyword for future implementation
    MISSING_FILES_HANDLE: str = "missing_files_handle"
    EXCESSIVE_FILES_HANDLE: str = "excessive_files_handle"

    SUBSET_LABEL: str = "subset_label"
    PATH_DATA: str = "path_data"


@dataclass(frozen=True)
class DatasetTypeKW:
    """
    Used to specify the datatype in the display_selected_transfoms
    """
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"

@dataclass(frozen=True)
class FileMissingHandleKW:
    """
    Keywords used to specify the handling of missing files
    """
    SKIP = "skip"
    RAISE = "raise"
    IGNORE = "ignore"

@dataclass(frozen=True)
class FileExcessiveHandleKW:
    """
    Keywords used to specify the handling of excessive files
    """
    USE_FIRST_AND_WARN = "use_first_and_warn"


