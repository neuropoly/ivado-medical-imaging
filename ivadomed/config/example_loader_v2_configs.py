from __future__ import annotations

from copy import deepcopy

from ivadomed.keywords import DataloaderKW, FileMissingHandleKW, FileExcessiveHandleKW

"""
This is a centralized place to store all the example config files for the loader.
Purpose is to provide an updated view of how the dictionary structures can be used to reflect latest examples
"""

from testing.common_testing_util import path_temp

path_mock_data = path_temp

example_file_dataset_group_config_json = {
            DataloaderKW.DATASET_GROUP_LABEL: "FileDataSetGroup1",
            DataloaderKW.TYPE: "FILES",
            DataloaderKW.PATH_DATA: path_mock_data,
            DataloaderKW.EXPECTED_INPUT: 2,
            DataloaderKW.EXPECTED_GT: 1,
            DataloaderKW.TRAINING: {
                    DataloaderKW.INPUT_GT: [
                        [["sub-01/ses-01/anat/sub-01_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-01/anat/sub-01_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-01/anat/sub-01_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-02/anat/sub-01_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-02/anat/sub-01_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-02/anat/sub-01_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-01/ses-03/anat/sub-01_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-01/ses-03/anat/sub-01_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-01/ses-04/anat/sub-01_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-01/ses-04/anat/sub-01_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
            },
            DataloaderKW.VALIDATION: {
                    DataloaderKW.INPUT_GT: [
                        [["sub-02/ses-01/anat/sub-02_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-01/anat/sub-02_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-01/anat/sub-02_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-02/anat/sub-02_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-02/anat/sub-02_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-02/anat/sub-02_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-02/ses-03/anat/sub-02_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-02/ses-03/anat/sub-02_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-02/ses-04/anat/sub-02_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-02/ses-04/anat/sub-02_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
            },
            DataloaderKW.TESTING: {
                    DataloaderKW.INPUT_GT:  [
                        [["sub-03/ses-01/anat/sub-03_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-01/anat/sub-03_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-01/anat/sub-03_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-02/anat/sub-03_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-02/anat/sub-03_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-02/anat/sub-03_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-03/ses-03/anat/sub-03_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-03/ses-03/anat/sub-03_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-03/ses-04/anat/sub-03_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-03/ses-04/anat/sub-03_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
            },
}

example_2i1o_all_dataset_groups_config_json: dict = {
    DataloaderKW.DATASET_GROUPS: [
        example_file_dataset_group_config_json,
        {
            DataloaderKW.DATASET_GROUP_LABEL: "FileDataSetGroup2",
            DataloaderKW.TYPE: "FILES",
            DataloaderKW.PATH_DATA: path_mock_data,
            DataloaderKW.EXPECTED_INPUT: 2,
            DataloaderKW.EXPECTED_GT: 1,
            DataloaderKW.TRAINING: {
                    DataloaderKW.INPUT_GT: [
                        [["sub-04/ses-01/anat/sub-04_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-01/anat/sub-04_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-01/anat/sub-04_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-02/anat/sub-04_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-02/anat/sub-04_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-02/anat/sub-04_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-04/ses-03/anat/sub-04_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-04/ses-03/anat/sub-04_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-04/ses-04/anat/sub-04_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-04/ses-04/anat/sub-04_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
            },
            DataloaderKW.VALIDATION: {
                    DataloaderKW.INPUT_GT: [
                        [["sub-05/ses-01/anat/sub-05_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-01/anat/sub-05_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-01/anat/sub-05_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-02/anat/sub-05_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-02/anat/sub-05_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-02/anat/sub-05_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-05/ses-03/anat/sub-05_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-05/ses-03/anat/sub-05_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-05/ses-04/anat/sub-05_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-05/ses-04/anat/sub-05_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
            },
            DataloaderKW.TESTING: {
                    DataloaderKW.INPUT_GT: [
                        [["sub-06/ses-01/anat/sub-06_ses-01_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-01/anat/sub-06_ses-01_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-01/anat/sub-06_ses-01_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-02/anat/sub-06_ses-02_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-02/anat/sub-06_ses-02_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-02/anat/sub-06_ses-02_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-off_MTS.nii",
                          "sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-on_MTS.nii",
                          "sub-06/ses-03/anat/sub-06_ses-03_flip-1_mt-on_MTS.nii"],
                         ["derivatives/labels/sub-06/ses-03/anat/sub-06_ses-03_mt-off_MTS_lesion-manual-rater1.nii"]],
                        [["sub-06/ses-04/anat/sub-06_ses-04_flip-1_mt-off_MTS.nii", ],
                         ["derivatives/labels/sub-06/ses-04/anat/sub-06_ses-04_mt-off_MTS_lesion-manual-rater1.nii"]],
                    ],
                    DataloaderKW.MISSING_FILES_HANDLE: FileMissingHandleKW.SKIP,
                    DataloaderKW.EXCESSIVE_FILES_HANDLE: FileExcessiveHandleKW.USE_FIRST_AND_WARN,
            },
        },
    ],
    DataloaderKW.EXPECTED_INPUT: 2,
    DataloaderKW.EXPECTED_GT: 1,
}

# Create the example json after patchingg the 2i1o example
example_1i1o_all_dataset_groups_config_json: dict = deepcopy(example_2i1o_all_dataset_groups_config_json)
example_1i1o_all_dataset_groups_config_json.update(
    {
        DataloaderKW.EXPECTED_INPUT: 1,
        DataloaderKW.EXPECTED_GT: 1,
    }
)
example_1i1o_all_dataset_groups_config_json[DataloaderKW.DATASET_GROUPS][0].update(
    {
        DataloaderKW.EXPECTED_INPUT: 1,
        DataloaderKW.EXPECTED_GT: 1,
    }
)
example_1i1o_all_dataset_groups_config_json[DataloaderKW.DATASET_GROUPS][1].update(
    {
        DataloaderKW.EXPECTED_INPUT: 1,
        DataloaderKW.EXPECTED_GT: 1,
    }
)

if __name__=="__main__":

    import pprint

    pprint.pprint(example_2i1o_all_dataset_groups_config_json)