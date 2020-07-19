#!/usr/bin/env python

import os
import argparse
import numpy as np
from collections import defaultdict
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True,
                        help="Input log directory.")
    parser.add_argument("-o", "--output", required=False,
                        help="Output folder. If not specified, results are saved under "
                             "input_folder/plot_training_curves.")
    return parser


def find_events(input_folder):
    """Get TF events path from input_folder.

    Args:
        input_folder (str): Input folder path.
    Returns:
        dict: keys are subfolder names and values are events' paths.
    """
    dict = {}
    for fold in os.listdir(input_folder):
        fold_path = os.path.join(input_folder, fold)
        if os.path.isdir(fold_path):
            event_list = [f for f in os.listdir(fold_path) if f.startswith("events.out.tfevents.")]
            if len(event_list):
                if len(event_list) > 1:
                    print('Multiple events found in this folder: {}.\nPlease keep only one before running '
                          'this script again.'.format(fold_path))
                dict[fold] = os.path.join(input_folder, fold, event_list[0])
    return dict


def get_data(event_dict):
    """Get data as Pandas dataframe.

    Args:
        event_dict (dict): Dictionary containing the TF event names and their paths.
    Returns:
        Pandas Dataframe: where the columns are the metrics or losses and the rows represent the epochs.
    """
    metrics = defaultdict(list)
    for tf_tag in event_dict:
        for e in summary_iterator(event_dict[tf_tag]):
            for v in e.summary.value:
                if isinstance(v.simple_value, float):
                    if tf_tag.startswith("Validation_Metrics_"):
                        tag = tf_tag.split("Validation_Metrics_")[1]
                    elif tf_tag.startswith("losses_"):
                        tag = tf_tag.split("losses_")[1]
                    else:
                        print("Unknown TF tag: {}.".format(tf_tag))
                        exit()
                    metrics[tag].append(v.simple_value)
    metrics_df = pd.DataFrame.from_dict(metrics)
    return metrics_df


def plot_curve(data, y_label, fname_out):
    """Plot curve of metrics or losses for each epoch.

    Args:
        data (pd.DataFrame):
        y_label (str): Label for the y-axis.
        fname_out (str): Save plot with this filename.
    """
    # Create count of the number of epochs
    epoch_count = range(1, len(data) + 1)

    for k in data.keys():
        plt.plot(epoch_count, data[k], 'r--')

    plt.legend(data.keys())
    plt.xlabel('Epoch')
    plt.ylabel(y_label)
    plt.show()


def run_plot_training_curves(input_folder, output_folder):
    """Utility function to XX.

    XX

    For example::

        ivadomed_XX

    XX

    .. image:: ../../images/XX
        :width: 600px
        :align: center

    Args:
         input_folder (string): Log directory name. Flag: --input, -i
    """
    # Find tf folders
    events_dict = find_events(input_folder)

    # Get data as dataframe
    events_vals_df = get_data(events_dict)

    # Create output folder
    if output_folder is None:
        output_folder = os.path.join(input_folder, "plot_training_curves")
    if os.path.isdir(output_folder):
        print("Output folder already exists: {}.".format(output_folder))
    else:
        print("Creating output folder: {}.".format(output_folder))

    # Plot train and valid losses together
    fname_out = os.path.join(output_folder, "losses.png")
    loss_keys = [k for k in events_vals_df.keys() if k.endswith("loss")]
    plot_curve(events_vals_df[loss_keys], "loss", fname_out)

    # Plot each validation metric separetly
    for tag in events_vals_df.keys():
        if not tag.endswith("loss"):
            fname_out = os.path.join(output_folder, tag+".png")
            plot_curve(events_vals_df[tag], tag, fname_out)


def main():
    parser = get_parser()
    args = parser.parse_args()
    input_folder = args.input
    output_folder = args.output
    # Run script
    run_plot_training_curves(input_folder, output_folder)


if __name__ == '__main__':
    main()
