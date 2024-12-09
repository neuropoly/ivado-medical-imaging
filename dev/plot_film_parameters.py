#!/usr/bin/env python
import sys
import os
import json
import operator
import functools

import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
import seaborn as sns

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from loguru import logger

from ivadomed import config_manager as imed_config_manager


def plot_histogram(data, layer_no, fname_out):
    """
    Save the histograms showing the frequency of the values inside gammas or betas tensors for one layer.

    :param data: input data, which are gammas or betas for one layer
    :param layer_no: number of the layer to consider
    :param fname_out: directory to save the figure
    """
    bins = np.linspace(0, 1, 100)

    fig = plt.figure(figsize=(12, 12))
    plt.title(f'Histogram: Layer {layer_no}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')

    # Flatten data
    data = np.array(functools.reduce(operator.iconcat, data, [])).ravel()

    plt.hist(data, bins)
    fig.savefig(fname_out)


def visualize_pca(data, metadata_values, layer_no, fname_out):
    """
    Save the PCA graphs showing gammas or betas tensors for one layer.

    :param data: input data, which are gammas or betas for one layer
    :param metadata_values: numpy array with the metadata values of all the images
    :param layer_no: number of the layer to consider
    :param fname_out: directory to save the figure
    """
    pca_df = pd.DataFrame()

    pca = PCA(n_components=2)

    # Dim 0 will be the number of examples
    data = np.concatenate(list(data), axis=0)
    metadata_values = np.array(functools.reduce(operator.iconcat, metadata_values, [])).ravel()

    pca_result = pca.fit_transform(data)
    pca_df2 = pd.DataFrame()
    pca_df2['pca1'] = pca_result[:, 0]
    pca_df2['pca2'] = pca_result[:, 1]
    pca_df2['contrast'] = metadata_values
    pca_df = pd.concat([pca_df, pca_df2])

    # Create the graph and save it
    fig = plt.figure(figsize=(16, 10))
    plt.title(f"PCA: Layer {layer_no}")
    sns.scatterplot(
        x="pca1", y="pca2",
        hue="contrast",
        data=pca_df,
        legend="full",
        alpha=1)
    fig.savefig(fname_out)


def visualize_tsne(data, metadata_values, layer_no, fname_out):
    """
    Save the t-SNE graphs showing gammas or betas tensors for one layer.

    :param data: input data, which are gammas or betas for one layer
    :param metadata_values: numpy array with the metadata values of all the images
    :param layer_no: number of the layer to consider
    :param fname_out: directory to save the figure
    """
    tsne_df = pd.DataFrame()

    tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)

    # Dim 0 will be the number of examples
    data = np.concatenate(list(data), axis=0)
    contrast_images = np.array(functools.reduce(operator.iconcat, metadata_values, [])).ravel()

    tsne_results = tsne.fit_transform(data)
    tsne_df2 = pd.DataFrame()

    tsne_df2['tsne-2d-one'] = tsne_results[:, 0]
    tsne_df2['tsne-2d-two'] = tsne_results[:, 1]
    tsne_df2['contrast'] = contrast_images

    tsne_df = pd.concat([tsne_df, tsne_df2])

    logger.info("t-SNE done!")

    # Visualize
    fig = plt.figure(figsize=(16,10))
    plt.title(f"t-SNE: Layer {layer_no}")
    sns.scatterplot(
        x="tsne-2d-one",
        y="tsne-2d-two",
        hue="contrast",
        data=tsne_df,
        legend="full",
        alpha=1
    )
    fig.savefig(fname_out)


def run_main(context):
    """
    Main command to create and save the graphs to visualize the film parameters.

    :param context: this is a dictionary with all data from the
                    configuration file from which we only use:
                        - 'path_output': folder name where log files are saved
    """
    path_output = context["path_output"]

    gammas = {}
    betas = {}
    for i in range(1, 9):
        if np.load(path_output + f"/gamma_layer_{i}.npy", allow_pickle=True).size != 0:
            gammas[i] = np.load(path_output + f"/gamma_layer_{i}.npy", allow_pickle=True)
        if np.load(path_output + f"/beta_layer_{i}.npy", allow_pickle=True).size != 0:
            betas[i] = np.load(path_output + f"/beta_layer_{i}.npy", allow_pickle=True)

    metadata_values = np.load(path_output + "/metadata_values.npy", allow_pickle=True)

    out_dir = context["path_output"] + "/film-parameters-visualization"
    if not os.path.isdir(out_dir):
        os.makedirs(out_dir)

    # save histograms with gammas and betas values
    for layer_no in gammas.keys():
        plot_histogram(gammas[layer_no], layer_no, out_dir + f"/hist_gamma_{layer_no}.png")
        plot_histogram(betas[layer_no], layer_no, out_dir + f"/hist_beta_{layer_no}.png")

    # save PCA for betas and gammas except for the last layer due to gammas/betas shapes
    for layer_no in gammas.keys():
        try:
            visualize_pca(gammas[layer_no], metadata_values, layer_no, out_dir + f"/pca_gamma_{layer_no}.png")
        except ValueError:
            logger.error(f"No PCA for gamma from the film layer {layer_no} because of a too small dimension.")
        try:
            visualize_pca(betas[layer_no], metadata_values, layer_no, out_dir + f"/pca_beta_{layer_no}.png")
        except ValueError:
            logger.error(f"No PCA for beta from the film layer {layer_no} because of a too small dimension.")

    # save tsne for betas and gammas
    for layer_no in gammas.keys():
        visualize_tsne(gammas[layer_no], metadata_values, layer_no, out_dir + f"/tsne_gamma_{layer_no}.png")
        visualize_tsne(betas[layer_no], metadata_values, layer_no, out_dir + f"/tsne_beta_{layer_no}.png")


if __name__ == "__main__":
    fname_config_file = sys.argv[1]

    context = imed_config_manager.ConfigurationManager(fname_config_file).get_config()

    run_main(context)
