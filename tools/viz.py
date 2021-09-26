import os
import numpy as np
import itertools
from matplotlib import pyplot as plt

# Source for distinct colors https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
colors_global = np.array([
    [230, 25, 75],
    [60, 180, 75],
    [255, 225, 25],
    [67, 99, 216],
    [245, 130, 49],
    [145, 30, 180],
    [70, 240, 240],
    [240, 50, 230],
    [188, 246, 12],
    [250, 190, 190],
    [0, 128, 128],
    [230, 190, 255],
    [154, 99, 36],
    [255, 250, 200],
    [128, 0, 0],
    [170, 255, 195],
    [128, 128, 0],
    [255, 216, 177],
    [0, 0, 117],
    [128, 128, 128],
    [255, 255, 255],
    [0, 0, 0]
]).astype(np.float)
colors_global = colors_global / 255.


def plot_histogram(pos_features, neg_features, pos_labels='Knowns', neg_labels='Unknowns', title="Histogram",
                   file_name='{}foo.pdf'):
    """
    This function plots the Histogram for Magnitudes of feature vectors.
    """
    pos_mag = np.sqrt(np.sum(np.square(pos_features), axis=1))
    neg_mag = np.sqrt(np.sum(np.square(neg_features), axis=1))

    pos_hist = np.histogram(pos_mag, bins=500)
    neg_hist = np.histogram(neg_mag, bins=500)

    fig, ax = plt.subplots(figsize=(4.5, 1.75))
    ax.plot(pos_hist[1][1:], pos_hist[0].astype(np.float16) / max(pos_hist[0]), label=pos_labels, color='g')
    ax.plot(neg_hist[1][1:], neg_hist[0].astype(np.float16) / max(neg_hist[0]), color='r', label=neg_labels)

    ax.tick_params(axis='both', which='major', labelsize=12)

    plt.xscale('log')
    plt.tight_layout()
    if title is not None:
        plt.title(title)
    plt.savefig(file_name.format('Hist', 'pdf'), bbox_inches='tight')
    plt.show()


def plotter_2D(
        pos_features,
        labels,
        neg_features=None,
        pos_labels='Knowns',
        neg_labels='Unknowns',
        title=None,
        file_name='foo.pdf',
        final=False,
        pred_weights=None,
        heat_map=False):
    plt.figure(figsize=[6, 6])

    if heat_map:
        min_x, max_x = np.min(pos_features[:, 0]), np.max(pos_features[:, 0])
        min_y, max_y = np.min(pos_features[:, 1]), np.max(pos_features[:, 1])
        x = np.linspace(min_x * 1.5, max_x * 1.5, 200)
        y = np.linspace(min_y * 1.5, max_y * 1.5, 200)
        pnts = list(itertools.chain(itertools.product(x, y)))
        pnts = np.array(pnts)

        e_ = np.exp(np.dot(pnts, pred_weights))
        e_ = e_ / np.sum(e_, axis=1)[:, None]
        res = np.max(e_, axis=1)

        plt.pcolor(x, y, np.array(res).reshape(200, 200).transpose(), rasterized=True)

    colors = colors_global
    if neg_features is not None:
        # Remove black color from knowns
        colors = colors_global[:-1, :]

    # TODO:The following code segment needs to be improved
    colors_with_repetition = colors.tolist()
    for i in range(int(len(set(labels.tolist())) / colors.shape[0])):
        colors_with_repetition.extend(colors.tolist())
    colors_with_repetition.extend(colors.tolist()[:int(colors.shape[0] % len(set(labels.tolist())))])
    colors_with_repetition = np.array(colors_with_repetition)

    labels_to_int = np.zeros(labels.shape[0])
    for i,l in enumerate(set(labels.tolist())):
        labels_to_int[labels==l]=i

    plt.scatter(pos_features[:, 0], pos_features[:, 1], c=colors_with_repetition[labels_to_int.astype(np.int)],
                edgecolors='none', s=5)
    if neg_features is not None:
        plt.scatter(neg_features[:, 0], neg_features[:, 1], c='k', edgecolors='none', s=15, marker="*")
    if final:
        plt.gca().spines['right'].set_position('zero')
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labeltop=False, labelleft=False,
                        labelright=False)
        plt.axis('equal')

    plt.savefig(file_name.format('2D_plot', 'png'), bbox_inches='tight')
    plt.show()
    if neg_features is not None:
        plot_histogram(pos_features, neg_features, pos_labels=pos_labels, neg_labels=neg_labels, title=title,
                       file_name=file_name.format('hist','pdf'))






def sigmoid_2D_plotter(
                        pos_features,
                        labels,
                        neg_features=None,
                        pos_labels='Knowns',
                        neg_labels='Unknowns',
                        title=None,
                        file_name='foo.pdf',
                        final=False,
                        pred_weights=None,
                        heat_map=False):
    plt.figure(figsize=[6, 6])

    if heat_map:
        min_x, max_x = np.min(pos_features[:, 0]), np.max(pos_features[:, 0])
        min_y, max_y = np.min(pos_features[:, 1]), np.max(pos_features[:, 1])
        x = np.linspace(min_x * 1.5, max_x * 1.5, 200)
        y = np.linspace(min_y * 1.5, max_y * 1.5, 200)
        pnts = list(itertools.chain(itertools.product(x, y)))
        pnts = np.array(pnts)

        e_ = np.exp(np.dot(pnts, pred_weights))
        e_ = e_ / np.sum(e_, axis=1)[:, None]
        res = np.max(e_, axis=1)

        plt.pcolor(x, y, np.array(res).reshape(200, 200).transpose(), rasterized=True)

    colors = colors_global
    if neg_features is not None:
        # Remove black color from knowns
        colors = colors_global[:-1, :]

    colors_with_repetition = colors.tolist()
    for i in range(10):
        plt.scatter(pos_features[labels==i, 0], pos_features[labels==i, 1], c=colors_with_repetition[i], edgecolors='none', s=1.-(i/10))
    if neg_features is not None:
        plt.scatter(neg_features[:, 0], neg_features[:, 1], c='k', edgecolors='none', s=15, marker="*")
    if final:
        plt.gca().spines['right'].set_position('zero')
        plt.gca().spines['bottom'].set_position('zero')
        plt.gca().spines['left'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.tick_params(axis='both', bottom=False, left=False, labelbottom=False, labeltop=False, labelleft=False,
                        labelright=False)
        plt.axis('equal')

    plt.savefig(file_name.format('2D_plot', 'png'), bbox_inches='tight')
    plt.show()
    if neg_features is not None:
        plot_histogram(pos_features, neg_features, pos_labels=pos_labels, neg_labels=neg_labels, title=title,
                       file_name=file_name.format('hist','pdf'))



def plot_OSRC(to_plot, no_of_false_positives=None, filename=None, title=None):
    """
    :param to_plot: list of tuples containing (knowns_accuracy,OSE,label_name)
    :param no_of_false_positives: To write on the x axis
    :param filename: filename to write
    :return: None
    """
    fig, ax = plt.subplots()
    if title != None:
        fig.suptitle(title, fontsize=20)
    for plot_no, (knowns_accuracy,OSE,label_name) in enumerate(to_plot):
        ax.plot(OSE, knowns_accuracy, label=label_name)
    ax.set_xscale('log')
    ax.autoscale(enable=True, axis='x', tight=True)
    ax.set_ylim([0, 1])
    ax.set_ylabel('Correct Classification Rate', fontsize=18, labelpad=10)
    if no_of_false_positives is not None:
        ax.set_xlabel(f"False Positive Rate : Total Unknowns {no_of_false_positives}", fontsize=18, labelpad=10)
    else:
        ax.set_xlabel(f"False Positive Rate", fontsize=18, labelpad=10)
    ax.legend(loc='lower center', bbox_to_anchor=(-1.25, 0.), ncol=1, fontsize=18, frameon=False)
    # ax.legend(loc="upper left")
    if filename is not None:
        fig.savefig(f"{filename}.pdf", bbox_inches="tight")
    plt.show()
