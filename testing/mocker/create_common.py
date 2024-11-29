import pprint
from abc import abstractmethod

import json
import multiprocessing
from pathlib import Path
from typing import List
from loguru import logger

from ivadomed.keywords import MockerKW


class CreateSubject(multiprocessing.Process):
    """
    This is s the abstract class that highlights the main way to interact with subject related creation process
    """

    def __init__(
        self,
        process_id,
        index_subject,
        path_to_mock_data,
        list_sessions,
        list_bids_details,
    ):
        """
        Constructor that focus on create all sessions/modalities for a given subject.
        :param process_id:
        :param index_subject:
        :param path_to_mock_data:
        :param list_sessions:
        :param list_bids_details:
        """
        super(CreateSubject, self).__init__()
        self.id = process_id
        self.index_subject = index_subject
        self.path_to_mock_data = path_to_mock_data
        self.list_sessions = list_sessions
        self.list_bids_details = list_bids_details

    @staticmethod
    @abstractmethod
    def default_constructor(path_to_mock_data: str):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        """
        Main entry point of the multiprocess.
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def create_one_json_nii_pair(self, session: str, bids_details: str):
        raise NotImplementedError

    @abstractmethod
    def generate_a_json_file(
        self,
        file_stem: str,
        session: str,
        bids_detail: str,
        bids_data_type: str,
    ):
        raise NotImplementedError

    def create_file_stem(self, session: str or None, bids_details: dict):
        """
        Create a file stem for a nifti/JSON image pairs.
        :param session:
        :param bids_details:
        :return:
        """

        if session:
            stem = f"sub-{self.index_subject:02d}_ses-{session:02d}"
        # No session spec
        else:
            stem = f"sub-{self.index_subject:02d}"

        # Append all possible bids keywords
        for keyword in bids_details.keys():
            if MockerKW.DATA_TYPE in keyword:
                stem += f"_{bids_details.get(MockerKW.DATA_TYPE, '')}"
            elif "LABELS" in keyword:
                stem += f"_{bids_details.get('LABELS', '')}"
            else:
                stem += f"_{keyword}-{bids_details.get(keyword,'')}"
        return stem

    def reject_missing_modalities(self, list_subject_specific_bids_dicts: List[dict]):
        """
        Validate the list of anatomy modalities.
        :param list_subject_specific_bids_dicts:
        :return:
        """
        for subject_specific_bids_dict in list_subject_specific_bids_dicts:
            if MockerKW.DATA_TYPE not in subject_specific_bids_dict:
                logger.critical(f"MODALITY is not present in the subject dictionary provided: {pprint.pformat(subject_specific_bids_dict)}")
                raise ValueError(
                    f"Missing MODALITY field in the provided subject_specific_bids_dict."
                )

    def reject_missing_labels(self, list_subject_specific_bids_dicts: List[dict]):
        """
        Validate the list of anatomy modalities.
        :param list_subject_specific_bids_dicts:
        :return:
        """
        for subject_specific_bids_dict in list_subject_specific_bids_dicts:
            if "LABELS" not in subject_specific_bids_dict:
                logger.critical(f"LABELS is not present in the subject dictionary provided: {pprint.pformat(subject_specific_bids_dict)}")
                raise ValueError(
                    f"Missing LABELS field in the provided subject_specific_bids_dict."
                )

    def generate_file_description_dict(self, bids_data_type: str) -> dict:
        """
        Fill a json dictionary with a description of the nifti image.
        :param bids_detail:
        :return:
        """
        path_current_file = Path(__file__).resolve().parent
        if bids_data_type == "anat":
            with open(f"{path_current_file}/example_mock_json/MRI-anatomy.json") as json_file:
                mock_meta_data = json.load(json_file)

        elif bids_data_type == "micr":
            with open(f"{path_current_file}/example_mock_json/microscopy.json") as json_file:
                mock_meta_data = json.load(json_file)

        elif bids_data_type == "dwi":
            logger.warning(f"{bids_data_type} json description generation not yet fully supported at the moment."
                           f" Caution advised to manually review the formualted placeholder diffusion JSON file for BIDS"
                           f" Compliance. ")
            with open(f"{path_current_file}/example_mock_json/MRI-diffusion.json") as json_file:
                mock_meta_data = json.load(json_file)

        elif bids_data_type == "func":
            logger.warning(f"{bids_data_type} json description generation not yet fully supported at the moment."
                           f" Caution advised to manually review the formualted placeholder diffusion JSON file for BIDS"
                           f" Compliance. ")
            with open(f"{path_current_file}/example_mock_json/task_events.json") as json_file:
                mock_meta_data = json.load(json_file)

        else:
            logger.critical(f"{bids_data_type} json description generation not yet fully supported at the moment."
                           f" Using default MRI-Common pattern description!")
            with open(f"{path_current_file}/example_mock_json/MRI-common.json") as json_file:
                mock_meta_data = json.load(json_file)

        return mock_meta_data

    def generate_dataset_description_dict(self) -> dict:
        """
        Fill a json dictionary with a description of the entire dataset of the collections of the nifti images.
        :return:
        """
        mock_meta_dataset = {
            "Name": "MOCK multi-permuted_bids_detail multi-session dataset",
            "BIDSVersion": "1.8.0",
            "Authors": "MOCK_AUTHOR",
            "GeneratedBy": {
                "Name": "MOCK ivadomed multi permuted_bids_detail multi session pipeline"
            },
        }
        return mock_meta_dataset
