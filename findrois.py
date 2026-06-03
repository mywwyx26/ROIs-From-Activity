import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform

# helper function: find best k by largest gap in merge distances
def best_k_from_linkage(lm, min_k=2, max_k=20):
    merge_distances = lm[:, 2]
    gaps = np.diff(merge_distances)

    # try gaps from largest to smallest, pick first that gives k > 1
    for idx in np.argsort(gaps)[::-1]:
        threshold = (merge_distances[idx] + merge_distances[idx + 1]) / 2
        trial_labels = fcluster(lm, t=threshold, criterion='distance') - 1
        k = len(np.unique(trial_labels))
        if min_k <= k <= max_k:
            return trial_labels, k
    
    # fallback
    labels = fcluster(lm, t=min_k, criterion='maxclust') - 1
    return labels, min_k

# main function
def findroi(data, cross_corr, filename, output='outputs'):
    # binarize cross corr image
    mean, stdev = np.mean(cross_corr), np.std(cross_corr)
    binarized = np.where(cross_corr > mean + stdev, 1, 0)
    labeled_array, num_features = ndimage.label(binarized)
    sizes = ndimage.sum(binarized, labeled_array, range(num_features + 1))

    # threshold for getting rid of small rois: 30 -> 72, 40 -> 63, 50 -> 54
    mask = sizes < 50
    remove_pixel = mask[labeled_array]
    labeled_array[remove_pixel] = 0
    labeled_array, num_features = ndimage.label(labeled_array)

    # get median fluorescence for each roi
    f0 = np.zeros(num_features) # all rois
    deltaf = np.zeros((num_features, data.shape[0]))
    for i in range(num_features):
        roi_trace = [ndimage.median(data[j], labels=labeled_array, index=i+1) for j in range(data.shape[0])]
        f0[i] = np.median(roi_trace)
        for j in range(data.shape[0]):
            deltaf[i, j] = roi_trace[j] - f0[i]

    # cross correlation and reordering
    corrcoef_matrix = np.corrcoef(deltaf)
    distance = 1 - corrcoef_matrix
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)
    linkage_matrix = linkage(squareform(distance), method='average')
    cluster_order = leaves_list(linkage_matrix)
    corrcoef_reordered = np.corrcoef(deltaf[cluster_order])
    
    # plot things and save them in output folder (claude coded)
    # split into 2 main groups first
    main_labels = fcluster(linkage_matrix, t=2, criterion='maxclust') - 1
    cluster_labels = np.zeros(num_features, dtype=int)
    next_label = 0

    for main_group in [0, 1]:
        idx = np.where(main_labels == main_group)[0]
        if len(idx) < 4:
            cluster_labels[idx] = next_label
            next_label += 1
            continue
        sub_corr = np.corrcoef(deltaf[idx])
        sub_dist = 1 - sub_corr
        sub_dist = (sub_dist + sub_dist.T) / 2
        np.fill_diagonal(sub_dist, 0)
        sub_linkage = linkage(squareform(sub_dist), method='average')
        sub_labels, sub_k = best_k_from_linkage(sub_linkage)
        for s in range(sub_k):
            cluster_labels[idx[sub_labels == s]] = next_label
            next_label += 1
    n_clusters = next_label

    # color coding
    set1 = [plt.cm.Set1(i) for i in range(9)]
    set2 = [plt.cm.Set2(i) for i in range(8)]
    set3 = [plt.cm.Set3(i) for i in range(12)]
    all_colors = set1 + set2 + set3  # 29 total
    colors = [all_colors[i % len(all_colors)] for i in range(n_clusters)]

    # average traces of each cluster
    plt.figure(1, figsize = (20,15))
    groups = [[] for _ in range(n_clusters)]
    for cluster_idx in range(n_clusters):
        roi_indices = np.where(cluster_labels == cluster_idx)[0]  # 0-based ROI indices
        plt.subplot(n_clusters, 1, cluster_idx + 1)
        for roi_idx in roi_indices:
            groups[cluster_idx].append(roi_idx + 1)
        if len(roi_indices) > 0:
            avg_trace = np.mean(deltaf[roi_indices], axis=0)
            plt.plot(avg_trace, color=colors[cluster_idx], linewidth=1.5)
            plt.ylabel(f'C{cluster_idx+1}\n(n={len(roi_indices)})', fontsize=7, rotation=0, labelpad=35)
        plt.tick_params(labelbottom=(cluster_idx == n_clusters - 1))
    plt.figure(1).savefig(os.path.join(output, f"{filename}_clusters_over_time.svg"))

    # new labels color coded by group
    group_colors = np.zeros((*labeled_array.shape, 3))  # RGB image

    for group_idx, group in enumerate(groups):
        color = colors[group_idx][:3]  # RGB instead of RGBA
        for roi in group:
            mask = labeled_array == roi
            group_colors[mask] = color

    plt.figure(2, figsize = (20,15))
    plt.subplot(1,2,1)
    plt.imshow(group_colors)

    # with the big labeled diagram of cross corr
    plt.subplot(1,2,2)
    plt.imshow(corrcoef_reordered, cmap='Spectral', vmin=-1, vmax=1)
    ticks = np.arange(num_features)
    labels = [str(cluster_order[i] + 1) for i in range(num_features)]  # +1 for 1-based ROI numbers
    plt.xticks(ticks, labels, rotation=90, fontsize=6)
    plt.yticks(ticks, labels, fontsize=6)
    plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.figure(2).savefig(os.path.join(output, f"{filename}_groups_and_matrix.svg"))    
