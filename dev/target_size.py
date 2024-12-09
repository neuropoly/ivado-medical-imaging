#!/usr/bin/env python
##############################################################
#
# This scripts compute the distribution of each lesion size,
# in vox and in mm3.
#
# Could work with tumour or SC etc.
#
# It filters using "target_suffix" and "contrast_test"
# from config file.
#
# Usage: python dev/target_size.py -c <config_file_path>
#
# Example: python dev/target_size.py -c config/config.json
#
##############################################################

import argparse
import json
import os

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import seaborn as sns
from scipy.ndimage import label, generate_binary_structure
from loguru import logger

from ivadomed import config_manager as imed_config_manager
from ivadomed import utils as imed_utils

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", help="Config file path.")

    return parser


def plot_distrib(arr, label, xlim, fname_out):
    fig = plt.figure()

    sns.distplot(arr, hist=False, kde=True, rug=True,
                 color='darkblue',
                 kde_kws={'linewidth': 2},
                 rug_kws={'color': 'black'})

    plt.xlabel(label)
    plt.xlim(xlim)
    plt.ylabel('Density')
    fig.savefig(fname_out)
    logger.info(f"\tSave as: {fname_out}")


def run_main(args):
    context = imed_config_manager.ConfigurationManager(args.c).get_config()

    path_folder = os.path.join(context['path_data'], 'derivatives', 'labels')

    bin_struct = generate_binary_structure(3, 2)  # 18-connectivity

    vox_lst, mm3_lst = [], []
    for s in os.listdir(path_folder):
        s_fold = os.path.join(path_folder, s, 'anat')
        if os.path.isdir(s_fold):
            for f in os.listdir(s_fold):
                c = f.split(s + '_')[-1].split(context["target_suffix"])[0]
                if f.endswith(context["target_suffix"] + '.nii.gz') and c in context["contrast_test"]:
                    f_path = os.path.join(s_fold, f)
                    im = nib.load(f_path)
                    data = np.asanyarray(im.dataobj)
                    px, py, pz = im.header['pixdim'][1:4]
                    del im

                    if np.any(data):
                        data_label, n = label(data,
                                              structure=bin_struct)
                        for idx in range(1, n + 1):
                            data_idx = (data_label == idx).astype(np.int)

                            n_vox = np.count_nonzero(data_idx)
                            vox_lst.append(n_vox)
                            mm3_lst.append(n_vox * px * py * pz)

    logger.debug("\nTarget distribution in vox:")
    imed_utils.print_stats(vox_lst)
    plot_distrib(vox_lst, context["target_suffix"] + ' size in vox',
                 [0, np.percentile(vox_lst, 90)],
                 context["target_suffix"] + '_vox.png')

    logger.info("\nTarget distribution in mm3:")
    imed_utils.print_stats(mm3_lst)
    plot_distrib(mm3_lst, context["target_suffix"] + ' size in mm3',
                 [0, np.percentile(mm3_lst, 90)],
                 context["target_suffix"] + '_mm3.png')


if __name__ == '__main__':
    parser = get_parser()
    arguments = parser.parse_args()
    run_main(arguments)
