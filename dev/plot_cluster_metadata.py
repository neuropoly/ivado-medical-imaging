#!/usr/bin/env python
# Usage:
#	python dev/plot_cluster_metadata.py <config_file>
# Example:
#	python dev/plot_cluster_metadata.py config/config_pr28.json

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib

from torchvision import transforms as torch_transforms
from loguru import logger

from ivadomed.loader.bids_dataset import BidsDataset
from ivadomed import config_manager as imed_config_manager
from ivadomed.loader.slice_filter import SliceFilter
from ivadomed import transforms as imed_transforms

metadata_type = ['FlipAngle', 'EchoTime', 'RepetitionTime']
metadata_range = {'FlipAngle': [0, 180, 0.5], 'EchoTime': [10 ** (-3), 10 ** (0), 10 ** (-3)],
                  'RepetitionTime': [10 ** (-3), 10 ** (1), 10 ** (-2)]}


def plot_decision_boundaries(data, model, x_range, metadata_name, fname_out):
    fig = plt.figure()

    x_min, x_max = x_range[0], x_range[1]
    y_min, y_max = 0, (x_max - x_min) * 0.2
    xx, yy = np.meshgrid(np.arange(x_min, x_max, x_range[2]),
                         np.arange(y_min, y_max, x_range[2]))

    Z = [model.predict(v) for v in xx.ravel()]
    Z = np.asarray(Z).reshape(xx.shape)

    plt.contourf(xx, yy, Z, cmap=plt.cm.jet, alpha=0.8)

    for s, y_val in zip(['train', 'valid', 'test'], [0.25, 0.5, 0.75]):
        plt.scatter(data[s][metadata_name], [(y_max - y_min) * y_val for v in data[s][metadata_name]], c='k')

    plt.xlabel(metadata_name)
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.yticks(())
    if metadata_name != 'FlipAngle':
        plt.xscale('log')

    fig.savefig(fname_out)
    logger.info(f"\tSave as: {fname_out}")


def run_main(context):
    no_transform = torch_transforms.Compose([
        imed_transforms.CenterCrop([128, 128]),
        imed_transforms.NumpyToTensor(),
        imed_transforms.NormalizeInstance(),
    ])

    out_dir = context["path_output"]
    split_dct = joblib.load(os.path.join(out_dir, "split_datasets.joblib"))
    metadata_dct = {}
    for subset in ['train', 'valid', 'test']:
        metadata_dct[subset] = {}
        ds = BidsDataset(context["path_data"],
                         subject_lst=split_dct[subset],
                         contrast_lst=context["contrast_train_validation"]
                         if subset != "test" else context["contrast_test"],
                         transform=no_transform,
                         slice_filter_fn=SliceFilter())

        for m in metadata_type:
            if m in metadata_dct:
                metadata_dct[subset][m] = [v for m_lst in [metadata_dct[subset][m], ds.metadata[m]] for v in m_lst]
            else:
                metadata_dct[subset][m] = ds.metadata[m]

    cluster_dct = joblib.load(os.path.join(out_dir, "clustering_models.joblib"))

    out_dir = os.path.join(out_dir, "cluster_metadata")
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    for m in metadata_type:
        values = [v for s in ['train', 'valid', 'test'] for v in metadata_dct[s][m]]
        logger.info(f"\n{m}: Min={min(values)}, Max={max(values)}, Median={np.median(values)}")
        plot_decision_boundaries(metadata_dct, cluster_dct[m], metadata_range[m], m, os.path.join(out_dir, m + '.png'))


if __name__ == "__main__":
    fname_config_file = sys.argv[1]

    context = imed_config_manager.ConfigurationManager(fname_config_file).get_config()

    run_main(context)
