'''
FINDROIS.PY
    This file takes the data and cross correlation image from readfiles.py and uses it to identify ROIs.
    Then the ROIs are sorted into clusters using hierarchical clustering, and silhouette score is used to
    find the best value of k (number of clusters). Each cluster's activity is plotted over time and the
    corresponding ROIs are color coded on the binarized image.

Inputs:
    data (3D ndarray): the .tif file as an array
    cross_corr (2D ndarray): the cross correlation image
    filename (str): the name of the file being analyzed, for .svg file naming
    output (str, default = "outputs"): the folder to save the .svg files to

Outputs:
    clusters_over_time (.svg): fluorescence of all ROIs in each cluster identified
    groups_and_matrix (.svg):
        groups: color coded by clusters version of the binarized cross corr image
        matrix: correlation coefficient matrix, in order from most to least correlated
'''

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_samples

# main function
def findroi(data, cross_corr, filename, output="outputs"):
    os.makedirs(output, exist_ok=True)

    # remove background: mask cross corr using sum of mean and std binarized images
    mean_img = np.mean(data, axis=0)
    std_img  = np.std(data, axis=0)
    sum_img  = mean_img + std_img
    bg_mask  = np.where(sum_img > np.mean(sum_img), 1, 0)
    cross_corr = cross_corr * bg_mask

    # binarize cross corr image: 80th percentile of foreground (non-zero) pixels
    fg_pixels = cross_corr[cross_corr > 0]
    binarized = np.where(cross_corr > np.percentile(fg_pixels, 80), 1, 0)
    labeled_array, num_features = ndimage.label(binarized)
    sizes = ndimage.sum(binarized, labeled_array, range(num_features + 1))

    # threshold for getting rid of small rois: 50px
    mask = sizes < 50
    remove_pixel = mask[labeled_array]
    labeled_array[remove_pixel] = 0
    labeled_array, num_features = ndimage.label(labeled_array)

    # get median fluorescence for each roi
    f0 = np.zeros(num_features)
    deltaf = np.zeros((num_features, data.shape[0]))
    for i in range(num_features):
        roi_trace = [ndimage.median(data[j], labels=labeled_array, index=i+1) for j in range(data.shape[0])]
        f0[i] = np.median(roi_trace)
        for j in range(data.shape[0]):
            deltaf[i, j] = (roi_trace[j] - f0[i]) / f0[i]

    # corrcoef and distance matrices
    corrcoef_matrix = np.corrcoef(deltaf)
    distance = 1 - corrcoef_matrix
    distance = (distance + distance.T) / 2
    np.fill_diagonal(distance, 0)

    # hierarchical clustering
    linkage_matrix = linkage(squareform(distance), method='average')
    cluster_order = leaves_list(linkage_matrix)
    corrcoef_reordered = np.corrcoef(deltaf[cluster_order])

    # find best k using mean silhouette score
    min_clusters = 2
    best_k, best_score, best_raw_labels = min_clusters, -1, None
    all_scores = {}

    for k in range(min_clusters, int(min(20, np.ceil(num_features / 2)))):
        labels = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_samples(distance, labels, metric='precomputed').mean()
        all_scores[k] = score

    sorted_ks = sorted(all_scores.keys())

    # Return first k that is higher than window neighbors on each side.
    def find_peak(scores, ks, window):
        for i in range(window, len(ks) - window):
            k = ks[i]
            s = scores[k]
            neighbors = [scores[ks[i + d]] for d in range(-window, window + 1) if d != 0]
            if all(s > n for n in neighbors):
                return k
        return None

    best_k = (
        find_peak(all_scores, sorted_ks, window=2) or
        find_peak(all_scores, sorted_ks, window=1) or
        max(all_scores, key=all_scores.get)
    )

    best_raw_labels = fcluster(linkage_matrix, t=best_k, criterion='maxclust') - 1
    best_score = all_scores[best_k]

    # flag noise: ROIs with negative per-ROI silhouette score
    noise_threshold = 0.0
    per_roi_scores = silhouette_samples(distance, best_raw_labels, metric='precomputed')
    is_noise = per_roi_scores < noise_threshold

    # build final labels: real cluster labels for good ROIs, individual labels for noise
    n_real = len(np.unique(best_raw_labels))
    cluster_labels = best_raw_labels.copy()
    next_noise_label = n_real
    for i in range(num_features):
        if is_noise[i]:
            cluster_labels[i] = next_noise_label
            next_noise_label += 1
    n_clusters = next_noise_label
    noise_count = int(is_noise.sum())

    # sort real clusters by ROI count descending and remap labels accordingly
    real_cluster_sizes = [(i, len(np.where(cluster_labels == i)[0])) for i in range(n_real)]
    real_cluster_sizes.sort(key=lambda x: x[1], reverse=True)

    label_remap = {}
    for new_idx, (old_idx, _) in enumerate(real_cluster_sizes):
        label_remap[old_idx] = new_idx
    # noise labels stay as-is (they're already >= n_real)
    cluster_labels = np.array([label_remap.get(l, l) for l in cluster_labels])

    # color coding: noise ROIs get gray
    set1 = [plt.cm.Set1(i) for i in range(8)] # has 9 but last one is gray so don't use it
    set3 = [plt.cm.Set3(i) for i in range(12)] # light gray is fine
    all_colors = set1 + set3
    colors = []
    for i in range(n_clusters):
        if i < n_real:
            colors.append(all_colors[i % len(all_colors)])
        else:
            colors.append((0.3, 0.3, 0.3, 1.0))  # gray for noise

    print(f"\nClustering results for {filename}:")
    print(f"  best k={best_k} (mean silhouette={best_score:.4f})")
    print(f"  clusters: {n_real}  |  noise ROIs: {noise_count}  |  total: {num_features}")

    # average traces of each cluster — noise excluded, real clusters already sorted by size
    groups = [[] for _ in range(n_clusters)]
    for cluster_idx in range(n_clusters):
        roi_indices = np.where(cluster_labels == cluster_idx)[0]
        for roi_idx in roi_indices:
            groups[cluster_idx].append(roi_idx + 1)

    plt.figure(1, figsize=(20, 15))
    for plot_idx in range(n_real):
        roi_indices = np.where(cluster_labels == plot_idx)[0]
        plt.subplot(n_real, 1, plot_idx + 1)
        if len(roi_indices) > 0:
            avg_trace = np.mean(deltaf[roi_indices], axis=0)
            plt.plot(avg_trace, color=colors[plot_idx], linewidth=1.5)
            plt.ylabel(f'C{plot_idx+1}\n(n={len(roi_indices)})', fontsize=7, rotation=0, labelpad=35)
        plt.tick_params(labelbottom=(plot_idx == n_real - 1))
    plt.figure(1).savefig(os.path.join(output, f"{filename}_clusters_over_time.svg"))
    plt.close(plt.figure(1))

    # color coded spatial map
    group_colors = np.zeros((*labeled_array.shape, 3))
    for group_idx, group in enumerate(groups):
        color = colors[group_idx][:3]
        for roi in group:
            mask = labeled_array == roi
            group_colors[mask] = color

    plt.figure(2, figsize=(20, 15))
    plt.subplot(1, 2, 1)
    plt.imshow(group_colors)
    plt.title("Clusters (gray=noise)")

    plt.subplot(1, 2, 2)
    plt.imshow(corrcoef_reordered, cmap='Spectral', vmin=-1, vmax=1)
    ticks = np.arange(num_features)
    tick_labels = [str(cluster_order[i] + 1) for i in range(num_features)]
    plt.xticks(ticks, tick_labels, rotation=90, fontsize=6)
    plt.yticks(ticks, tick_labels, fontsize=6)
    plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.figure(2).savefig(os.path.join(output, f"{filename}_groups_and_matrix.svg"))
    plt.close(plt.figure(2))

'''
Takes the data.npy and cross_corr.npy files from each subfolder in readfiles and outputs the resulting
graphs to the outputs folder. Useful to save time because the cross correlation function in readfiles.py
is very slow. Requires readfiles.py to be run first, so that there are existing .npy files.
'''
if __name__ == "__main__":
    print('FINDROIS only')
    subfolders = [f.path for f in os.scandir("readfiles") if f.is_dir()]

    for foldername in subfolders:
        output = os.path.join("outputs", os.path.basename(foldername))
        os.makedirs(output, exist_ok=True)

        data = np.load(f'{foldername}/data.npy')
        cross_corr = np.load(f'{foldername}/cross_corr.npy')
        findroi(data, cross_corr, filename=os.path.basename(foldername), output=output)
        print(f'findrois done: {output}')