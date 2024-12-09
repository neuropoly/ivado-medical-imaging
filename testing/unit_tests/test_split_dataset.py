import csv
import json
import pytest
import numpy as np
import pandas as pd
from ivadomed.loader import utils as imed_loader_utils
from testing.unit_tests.t_utils import create_tmp_dir, __tmp_dir__
from testing.common_testing_util import remove_tmp_dir
from pathlib import Path

PATH_DATA = Path(__tmp_dir__, 'bids')
PATH_LOG = Path(__tmp_dir__, 'log')
N = 200
N_CENTERS = 5


def setup_function():
    create_tmp_dir()


def load_dataset(split_params):
    patient_mapping = create_tsvfile()
    create_jsonfile()

    # Create log path
    if not PATH_LOG.is_dir():
        PATH_LOG.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(Path(PATH_DATA, "participants.tsv"), sep='\t')
    df['filename'] = df["participant_id"]
    train, val, test = imed_loader_utils.get_subdatasets_subject_files_list(split_params, df, str(PATH_LOG))
    return train, val, test, patient_mapping


@pytest.mark.parametrize('split_params', [{
    "fname_split": None,
    "random_seed": 6,
    "split_method": "participant_id",
    "data_testing": {"data_type": "institution_id", "data_value": [0]},
    "train_fraction": 0.6,
    "test_fraction": 0.2
}])
def test_per_center_testcenter_0(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    # Verify split proportion
    assert len(train) == round(0.6 * N)

    # Verify there is only the test center selected
    for sub in test:
        assert patient_mapping[sub]['center'] == '0'


@pytest.mark.parametrize('split_params', [{
    "fname_split": None,
    "random_seed": 6,
    "split_method": "participant_id",
    "data_testing": {"data_type": "institution_id", "data_value": []},
    "train_fraction": 0.2,
    "test_fraction": 0.4
}])
def test_per_center_without_testcenter(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    test_centers = set()
    for sub in test:
        test_centers.add(patient_mapping[sub]['center'])

    training_centers = set()
    for sub in train:
        training_centers.add(patient_mapping[sub]['center'])

    # Verify the test center proportion
    assert len(test_centers) == round(N_CENTERS * 0.4)

    # Verify test and training centers are fully different
    for train_center in training_centers:
        assert train_center not in test_centers


@pytest.mark.parametrize('split_params', [{
    "fname_split": None,
    "random_seed": 6,
    "split_method": "participant_id",
    "data_testing": {"data_type": None, "data_value": []},
    "train_fraction": 0.45,
    "test_fraction": 0.35
}])
def test_per_patient(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    assert np.isclose(len(train), round(N * 0.45), atol=1)
    assert np.isclose(len(test), round(N * 0.35), atol=1)


@pytest.mark.parametrize('split_params', [{
    "fname_split": None,
    "random_seed": 6,
    "split_method": "participant_id",
    "data_testing": {"data_type": None, "data_value": []},
    "train_fraction": 0.6,
    "test_fraction": 0
}])
def test_per_patient_2(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    assert np.isclose(len(train), round(N * 0.6), atol=1)
    assert np.isclose(len(val), round(N * 0.4), atol=1)
    assert np.isclose(len(test), 0, atol=1)


def check_balance(train, val, test, patient_mapping):
    for dataset in [train, val, test]:
        disability_count = {'0': 0, '1': 0, '2': 0}
        for sub in dataset:
            disability_count[patient_mapping[sub]['disability']] += 1

        assert np.isclose(disability_count['0'], disability_count['1'], atol=1)
        assert np.isclose(disability_count['1'], disability_count['2'], atol=1)
        assert np.isclose(disability_count['0'], disability_count['2'], atol=1)


@pytest.mark.parametrize('split_params', [{
    "fname_split": None,
    "random_seed": 6,
    "center_test": [],
    "balance": "disability",
    "split_method": "participant_id",
    "data_testing": {"data_type": None, "data_value": []},
    "train_fraction": 0.45,
    "test_fraction": 0.35
}])
def test_per_patient_balance(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    assert np.isclose(len(train), round(N * 0.45), atol=1)
    assert np.isclose(len(test), round(N * 0.35), atol=1)
    check_balance(train, val, test, patient_mapping)


@pytest.mark.parametrize('split_params', [{
    "fname_split": None,
    "random_seed": 6,
    "balance": "disability",
    "split_method": "participant_id",
    "data_testing": {"data_type": "institution_id", "data_value": [0]},
    "train_fraction": 0.4,
    "test_fraction": 0.2
}])
def test_per_center_balance(split_params):
    train, val, test, patient_mapping = load_dataset(split_params)

    # Verify split proportion
    assert np.isclose(len(train), round(0.4 * N), atol=2)

    # Verify there is only the test center selected
    for sub in test:
        assert patient_mapping[sub]['center'] == '0'

    check_balance(train, val, test, patient_mapping)


def create_tsvfile():
    # Create data path
    if not PATH_DATA.is_dir():
        PATH_DATA.mkdir(parents=True, exist_ok=True)

    patient_mapping = {}

    # Create participants.tsv with n participants
    participants = []
    for participant_id in range(N):
        row_participants = []
        patient_id = 'sub-00' + str(participant_id)
        row_participants.append(patient_id)
        # 3 different disabilities: 0, 1, or 2
        disability_id = str(participant_id % 3)
        row_participants.append(disability_id)
        # N_CENTERS different centers: 0, 1, ..., or N_CENTERS
        center_id = str(participant_id % N_CENTERS)
        row_participants.append(center_id)
        patient_mapping[patient_id] = {}
        patient_mapping[patient_id]['disability'] = disability_id
        patient_mapping[patient_id]['center'] = center_id
        participants.append(row_participants)

    with Path(PATH_DATA, "participants.tsv").open(mode='w') as tsv_file:
        tsv_writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        tsv_writer.writerow(["participant_id", "disability", "institution_id"])
        for item in sorted(participants):
            tsv_writer.writerow(item)

    return patient_mapping


def create_jsonfile():
    """Create dataset_description.json."""

    dataset_description = {}
    dataset_description[u'Name'] = 'Test'
    dataset_description[u'BIDSVersion'] = '1.2.1'

    with Path(PATH_DATA, "dataset_description.json").open(mode='w') as outfile:
        outfile.write(json.dumps(dataset_description, indent=2, sort_keys=True))
        outfile.close()


def teardown_function():
    remove_tmp_dir()
