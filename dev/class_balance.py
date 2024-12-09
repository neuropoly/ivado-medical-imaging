#!/usr/bin/env python
##############################################################
#
# This scripts compute the class balance in a given dataset.
#
# Usage: python dev/class_balance.py -c <config_file_path>
#
# Example: python class_balance.py -c config/config.json
#
# Contributors: charley
#
##############################################################

import argparse
import numpy as np

from ivadomed.loader.bids_dataset import BidsDataset
from ivadomed import config_manager as imed_config_manager
from ivadomed.loader import utils as imed_loader_utils
from ivadomed.loader.slice_filter import SliceFilter
from ivadomed import transforms as imed_transforms
from ivadomed import utils as imed_utils

from torchvision import transforms as torch_transforms
from torch.utils.data import DataLoader

from loguru import logger


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Config file path.")

    return parser


def run_main(args):
    context = imed_config_manager.ConfigurationManager(args.c).get_config()

    transform_lst = torch_transforms.Compose([
        imed_transforms.Resample(wspace=0.75, hspace=0.75),
        imed_transforms.CenterCrop([128, 128]),
        imed_transforms.NumpyToTensor(),
        imed_transforms.NormalizeInstance(),
    ])

    train_lst, valid_lst, test_lst = imed_loader_utils.split_dataset(context["path_data"],
                                                                     context["center_test"],
                                                                     context["split_method"],
                                                                     context["random_seed"])

    balance_dct = {}
    for ds_lst, ds_name in zip([train_lst, valid_lst, test_lst], ['train', 'valid', 'test']):
        logger.info(f"\nLoading {ds_name} set.\n")
        ds = BidsDataset(context["path_data"],
                         subject_lst=ds_lst,
                         target_suffix=context["target_suffix"],
                         contrast_lst=context["contrast_test"] if ds_name == 'test'
                         else context["contrast_train_validation"],
                         metadata_choice=context["metadata"],
                         contrast_balance=context["contrast_balance"],
                         transform=transform_lst,
                         slice_filter_fn=SliceFilter())

        logger.info(f"Loaded {len(ds)} axial slices for the {ds_name} set.")
        ds_loader = DataLoader(ds, batch_size=1,
                               shuffle=False, pin_memory=False,
                               collate_fn=imed_loader_utils.imed_collate,
                               num_workers=1)

        balance_lst = []
        for i, batch in enumerate(ds_loader):
            gt_sample = batch["gt"].numpy().astype(np.int)[0, 0, :, :]
            nb_ones = (gt_sample == 1).sum()
            nb_voxels = gt_sample.size
            balance_lst.append(nb_ones * 100.0 / nb_voxels)

        balance_dct[ds_name] = balance_lst

    for ds_name in balance_dct:
        logger.info(f"\nClass balance in {ds_name} set:")
        imed_utils.print_stats(balance_dct[ds_name])

    logger.info("\nClass balance in full set:")
    imed_utils.print_stats([e for d in balance_dct for e in balance_dct[d]])


if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)
