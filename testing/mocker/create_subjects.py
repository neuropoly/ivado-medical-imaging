import json
import itertools
from pathlib import Path
import multiprocessing
from typing import List

from nibabel import Nifti1Image
import numpy as np
import imageio
from ivadomed.keywords import MockerKW
from testing.mocker.create_common import CreateSubject
from testing.mocker.nifti_mocker import (
    create_mock_nifti1_object,
    create_mock_nifti2_object,
    check_nifty_data,
    save_nifty_data,
)


class CreateBIDSSubjects(CreateSubject):
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
        self.path_to_mock_data = Path(path_to_mock_data)
        self.list_sessions = list_sessions

        # Validate the list before proceeding
        self.reject_missing_modalities(list_bids_details)
        self.list_bids_details: List[dict] = list_bids_details

    @staticmethod
    def default_constructor(path_to_mock_data):
        """
        Default constructor.
        :return:
        """
        id = 1
        index_subject = [1, 2, 3, 4, 5]
        list_sessions = [1, 2, 3]
        list_modalities = ["T1w", "T2w", "FLAIR", "PD"]
        CreateBIDSSubjects(
            id, index_subject, path_to_mock_data, list_sessions, list_modalities
        )

    def run(self):
        """
        Main entry point of the multiprocess.
        :return:
        """
        self.create_sessions(self.list_sessions, self.list_bids_details)

        # Save mock_json dictionary into a json file:
        mock_json: dict = self.generate_dataset_description_dict()
        self.generate_modality_agnostic_samples_tsv()
        self.generate_modality_agnostic_participant_tsv()
        self.generate_readme()

        path_json_file: Path = self.path_to_mock_data / "dataset_description.json"
        with open(str(path_json_file), "w") as f:
            json.dump(mock_json, f, indent=4, sort_keys=True)

    def create_sessions(
        self, sessions: List[str] or None, list_bids_details: List[dict]
    ):
        # No session lists, single session
        if sessions is None:
            self.create_session_specific_modality_suffix(None, list_bids_details)

        # Session lists
        else:
            for session in sessions:
                self.create_session_specific_modality_suffix(session, list_bids_details)

    def create_session_specific_modality_suffix(
        self, session: str or None, list_bids_details_in_lists: List[dict]
    ):
        """
        Create the expected nifti files for a given session.
        :param session:
        :param list_bids_details_in_lists:
        :return:
        """
        # Take a single bids_deetail dict from the list.
        # This dictionary contain LISTS
        for a_bids_detail_dict in list_bids_details_in_lists:

            # Break down a single bids_detail_dictionary into concretized individual specific bids parameter pairs.
            # This is because bids_details could have list PER modalities.
            # e.g. acq = ["T1w", "T2w", "FLAIR", "PD"] needs to be broken down into FOR separate calls.
            # Source Inspiration: https://stackoverflow.com/a/61335465

            bids_standard_keys, bids_values = zip(*a_bids_detail_dict.items())

            list_permuted_dicts: List[dict] = [
                dict(zip(bids_standard_keys, a_set_of_permuted_values))
                for a_set_of_permuted_values in itertools.product(*bids_values)
            ]


            for a_permuted_bids_detail_dict in list_permuted_dicts:
                self.create_one_json_nii_pair(
                    session, a_permuted_bids_detail_dict
                )

    def create_one_json_nii_pair(self, session: str or None, permuted_bids_detail: dict):
        """
        Create a nifti/JSON image pairs with an explicitly set list of session and model
        :param session:
        :param permuted_bids_detail:
        :param kwargs:
        :return:
        """
        file_stem: str = self.create_file_stem(session, permuted_bids_detail)

        # default to anat if data type is somehow omitted (all bids MUST have a data type)
        bids_data_type = permuted_bids_detail.get(MockerKW.DATA_TYPE, "anat")

        self.generate_a_json_file(
            file_stem,
            session=session,
            bids_detail=permuted_bids_detail,
            bids_data_type=bids_data_type,
        )

        self.generate_a_nifti_file(
            file_stem,
            session=session,
            bids_data_type=bids_data_type
        )

    def create_one_json_image_pair(self, session: str or None, permuted_bids_detail: dict):
        """
        Create an image/JSON pairs with an explicitly set list of session and model
        :param session:
        :param permuted_bids_detail:
        :param kwargs:
        :return:
        """
        file_stem: str = self.create_file_stem(session, permuted_bids_detail)

        # default to anat if data type is somehow omitted (all bids MUST have a data type)
        bids_data_type = permuted_bids_detail.get(MockerKW.DATA_TYPE, "micr")

        self.generate_a_json_file(
            file_stem,
            session=session,
            bids_detail=permuted_bids_detail,
            bids_data_type=bids_data_type,
        )

        self.generate_an_image_file(
            file_stem,
            session=session,
            bids_data_type=bids_data_type
        )

    def generate_a_nifti_file(
        self, file_stem: str, bids_data_type: str, session: str or None
    ):
        """
        Produce the expected nifti files for a given bids_data_type and session.
        :param file_stem:
        :param bids_data_type:
        :param session:
        :return:
        """
        # Generate Nifti file
        file_name_nii: str = file_stem + ".nii"
        if session:
            # create nii file
            path_nii_file: Path = Path(
                self.path_to_mock_data,
                f"sub-{self.index_subject:02d}",
                f"ses-{session:02d}",
                f"{bids_data_type}",
                file_name_nii,
            )
        else:
            # create nii file
            path_nii_file: Path = Path(
                self.path_to_mock_data,
                f"sub-{self.index_subject:02d}",
                f"{bids_data_type}",
                file_name_nii,
            )
        mock_data: Nifti1Image = create_mock_nifti1_object()
        path_nii_file.parent.mkdir(parents=True, exist_ok=True)
        save_nifty_data(mock_data, path_nii_file)

    def generate_a_json_file(
        self,
        file_stem: str,
        session: str or None,
        bids_detail: dict,
        bids_data_type: str,
    ):
        """
        Generate a json file for a given bids_data_type and session.

        :param file_stem:
        :param bids_detail:
        :param session:
        :param bids_data_type:
        :return:
        """
        # Generate JSON file
        file_name_json: str = file_stem + ".json"

        if session:
            # path to JSON file
            path_json_file: Path = Path(
                self.path_to_mock_data,
                f"sub-{self.index_subject:02d}",
                f"ses-{session:02d}",
                f"{bids_data_type}",
                file_name_json,
            )
        else:
            # path to JSON file
            path_json_file: Path = Path(
                self.path_to_mock_data,
                f"sub-{self.index_subject:02d}",
                f"{bids_data_type}",
                file_name_json,
            )
        # ensure the parent folder exists
        path_json_file.parent.mkdir(parents=True, exist_ok=True)
        mock_json: object = self.generate_file_description_dict(bids_detail)

        # Save mock_json dictionary into a json file:
        with open(path_json_file, "w") as f:
            json.dump(mock_json, f, indent=4, sort_keys=True)

    def generate_modality_agnostic_samples_tsv(self):
        import csv
        with open(f'{self.path_to_mock_data}/samples.tsv', 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["a", "b", "c"])

    def generate_modality_agnostic_participant_tsv(self):
        import csv
        with open(f'{self.path_to_mock_data}/participants.tsv', 'w', newline='') as tsvfile:
            writer = csv.writer(tsvfile, delimiter='\t', lineterminator='\n')
            writer.writerow(["d", "e", "f"])

    def generate_readme(self):
        # Write an example README file
        readme_file: Path = Path(self.path_to_mock_data, "README.md")
        with open(readme_file, "w") as f:
            f.write(
                """
                # Mock BIDS data                
                \nThis is a mock BIDS dataset for testing purposes.                
                \n## Contributing
                \nIf you would like to contribute to this Mock dataset, please contact the Mock maintainer.
                \n## License
                \n[MIT](https://choosealicense.com/licenses/mit/)
                """
            )

    def generate_file_description_dict(self, bids_data_type: str) -> dict:
        pass

    def generate_an_image_file(self, file_stem, session, bids_data_type):
        """
        Create an image file with an explicitly set list of session and model
        Args:
            file_stem:
            session:
            bids_data_type:

        Returns:
        """
        # Generate a mock PNG file
        file_name_png: str = file_stem + ".png"
        if session:
            # path to JSON file
            path_image_file: Path = Path(
                self.path_to_mock_data,
                f"sub-{self.index_subject:02d}",
                f"ses-{session:02d}",
                f"{bids_data_type}",
                file_name_png,
            )
        else:
            # path to JSON file
            path_image_file: Path = Path(
                self.path_to_mock_data,
                f"sub-{self.index_subject:02d}",
                f"{bids_data_type}",
                file_name_png,
            )

        # Create a 100 by 100 pixel image require extensively

        img = np.zeros([100, 100, 3], dtype=np.uint8)
        img.fill(255)
        # Save img to disk
        path_image_file.parent.mkdir(parents=True, exist_ok=True)
        imageio.imwrite(path_image_file, img)


