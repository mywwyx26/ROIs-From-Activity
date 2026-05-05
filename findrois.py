# the imports have squigglies but it works if i do >python findrois.py
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, leaves_list
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def findroi(data, cross_corr, filename):
    # binarize cross corr image
    mean1, stdev1 = np.mean(cross_corr), np.std(cross_corr)
    binarized1 = np.where(cross_corr > mean1 + stdev1, 1, 0)
    labeled_array, num_features = ndimage.label(binarized1)
    sizes = ndimage.sum(binarized1, labeled_array, range(num_features + 1))

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

    # cross correlation and reordering (claude coded)
    corrcoef_matrix = np.corrcoef(deltaf)
    distance = 1 - corrcoef_matrix
    linkage_matrix = linkage(distance, method='average')
    cluster_order = leaves_list(linkage_matrix)  # new ordering of ROI indices
    deltaf_reordered = deltaf[cluster_order]
    corrcoef_reordered = np.corrcoef(deltaf_reordered)

    # plotting things and save them in output folder (claude coded)
    output = "outputs"
    os.makedirs(output, exist_ok = True)

    # attempt pca and kmeans to do grouping instead of manually, pca = 6 and k = 6
    scaler = StandardScaler().fit_transform(corrcoef_matrix)
    pca = PCA(n_components=6)
    pca_result = pca.fit_transform(scaler)
    kmeans = KMeans(n_clusters=6, init="k-means++", n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(pca_result)

    # average traces of each cluster (claude coded)
    n_clusters = 6
    plt.figure(1, figsize = (20,15))
    groups = [[] for _ in range(n_clusters)]
    for cluster_idx in range(n_clusters):
        roi_indices = np.where(cluster_labels == cluster_idx)[0]  # 0-based ROI indices
        plt.subplot(n_clusters, 1, cluster_idx + 1)
        for roi_idx in roi_indices:
            groups[cluster_idx].append(roi_idx + 1)
        if len(roi_indices) > 0:
            avg_trace = np.mean(deltaf[roi_indices], axis=0)
            plt.plot(avg_trace, color=plt.cm.Set1(cluster_idx), linewidth=1.5)
            plt.ylabel(f'C{cluster_idx+1}\n(n={len(roi_indices)})', fontsize=7, rotation=0, labelpad=35)
        plt.tick_params(labelbottom=(cluster_idx == n_clusters - 1))
    plt.figure(1).savefig(os.path.join(output, f"{filename}_clusters_over_time.svg"))

    # new labels color coded by group
    group_colors = np.zeros((*labeled_array.shape, 3))  # RGB image

    for group_idx, group in enumerate(groups):
        color = plt.cm.Set1(group_idx)[:3]  # RGB, same as plot 5
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
