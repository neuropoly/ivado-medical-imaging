import pytest
from typing import List

from ivadomed.keywords import MockerKW
from testing.functional_tests.t_utils import create_tmp_dir
from testing.mocker.create_derivatives import CreateBIDSDerivatives
from testing.mocker.create_subjects import CreateBIDSSubjects

def create_example_mock_bids_file_structures(path_temp: str):
    ######################
    # Subject Information
    ######################
    list_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    list_subject_sessions = [1, 2, 3, 4, 5]
    # This must be in the order of appending
    # MUST have bids_details keyword
    list_subject_specific_bids_dict = [
        {
            "acq": ["MTon", "MToff", "T1w"],  # Acquisition
            MockerKW.DATA_TYPE: ["MTS"],  # Modality
        },
        {
            "flip": [1, 2],  # Flip angle
            "mt": ["on", "off"],  # MT
            MockerKW.DATA_TYPE: ["MTS"],  # Modality
        },
        {
            "flip": [2],  # Flip angle
            "mt": ["off"],  # MT
            MockerKW.DATA_TYPE: ["MTS"],  # Modality
        }
    ]

    ######################
    # Derivative Information
    ######################
    list_derivatives_subjects = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

    list_derivatives_subject_sessions = [1, 2, 3, 4, 5]

    list_derivative_subject_specific_bids_dict = [
        {
            "mt": ["off"],  # MT
            MockerKW.DATA_TYPE: ["MTS"],  # Modality
            "LABELS": ["lesion-manual-rater1", "lesion-manual-rater2"]
        },
    ]

    create_mock_bids_file_structures(
        path_temp=path_temp,

        list_subjects=list_subjects,
        list_subject_sessions=list_subject_sessions,
        list_subject_specific_bids_dict=list_subject_specific_bids_dict,

        list_derivatives_subjects=list_derivatives_subjects,
        list_derivatives_subject_sessions=list_derivatives_subject_sessions,
        list_derivative_subject_specific_bids_dict=list_derivative_subject_specific_bids_dict
    )

def create_mock_bids_file_structures(
        path_temp: str,
        list_subjects: List[int],
        list_subject_sessions: List[int],
        list_subject_specific_bids_dict: List[dict],
        list_derivatives_subjects: List[int],
        list_derivatives_subject_sessions: List[int],
        list_derivative_subject_specific_bids_dict: List[dict]

):
    """
    Nothing strange, subjects, sessions, plans.
    :return:
    """
    # Ensure parent folder exists.
    create_tmp_dir(copy_data_testing_dir=False)

    path_mock_data = path_temp

    #######################
    # Subject Mock Section
    #######################
    # Create the subjects
    for index, subject in enumerate(list_subjects):
        bids_subject = CreateBIDSSubjects(
            process_id=index,
            index_subject=subject,
            path_to_mock_data=path_mock_data,
            list_sessions=list_subject_sessions,
            list_bids_details=list_subject_specific_bids_dict,
        )
        bids_subject.run()

    ##########################
    # Derivative Mock Section
    #########################
    # Create the derivatives
    for index, subject in enumerate(list_derivatives_subjects):
        bids_derivative = CreateBIDSDerivatives(
            process_id=index,
            index_subject=subject,
            path_to_mock_data=path_mock_data,
            list_sessions=list_derivatives_subject_sessions,
            list_bids_details=list_derivative_subject_specific_bids_dict,
        )
        bids_derivative.run()
