import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram
from scipy.spatial.distance import squareform
import hdbscan as hdbscan_pkg

# main function
def findroi(data, cross_corr, filename, output_dir="outputs"):
    output = output_dir
    # binarize cross corr image using Otsu's method
    from skimage.filters import threshold_otsu
    otsu_thresh = threshold_otsu(cross_corr)
    binarized = np.where(cross_corr > otsu_thresh, 1, 0)

    # comparison plot: mean+stdev vs otsu
    mean, stdev = np.mean(cross_corr), np.std(cross_corr)
    meanstd_thresh = mean + stdev
    binarized_meanstd = np.where(cross_corr > meanstd_thresh, 1, 0)

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Threshold comparison — {filename}", fontsize=11)

    axes[0, 0].imshow(cross_corr, cmap="gray")
    axes[0, 0].set_title("Original cross-correlation")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(binarized_meanstd, cmap="gray")
    axes[0, 1].set_title(f"mean+stdev  (thresh={meanstd_thresh:.3f})")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(binarized, cmap="gray")
    axes[0, 2].set_title(f"Otsu  (thresh={otsu_thresh:.3f})")
    axes[0, 2].axis("off")

    axes[1, 0].hist(cross_corr.ravel(), bins=100, color="steelblue", edgecolor="none")
    axes[1, 0].axvline(meanstd_thresh, color="tomato",    linewidth=1.5, label=f"mean+stdev ({meanstd_thresh:.3f})")
    axes[1, 0].axvline(otsu_thresh,    color="limegreen", linewidth=1.5, label=f"Otsu ({otsu_thresh:.3f})")
    axes[1, 0].set_title("Pixel intensity histogram")
    axes[1, 0].set_xlabel("Cross-correlation value")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend(fontsize=8)

    diff = binarized.astype(int) - binarized_meanstd.astype(int)
    diff_rgb = np.zeros((*diff.shape, 3))
    diff_rgb[diff ==  1] = [0, 1, 0]
    diff_rgb[diff == -1] = [1, 0, 0]
    axes[1, 1].imshow(diff_rgb)
    axes[1, 1].set_title("Difference (green=Otsu only, red=mean+stdev only)")
    axes[1, 1].axis("off")

    n_meanstd = int(binarized_meanstd.sum())
    n_otsu    = int(binarized.sum())
    axes[1, 2].bar(["mean+stdev", "Otsu"], [n_meanstd, n_otsu],
                   color=["tomato", "limegreen"], edgecolor="none")
    axes[1, 2].set_title("Foreground pixel count")
    axes[1, 2].set_ylabel("Pixels")
    for bar, val in zip(axes[1, 2].patches, [n_meanstd, n_otsu]):
        axes[1, 2].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 20,
                        str(val), ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    fig.savefig(os.path.join(output_dir, f"{filename}_threshold_comparison.svg"))
    plt.close(fig)

    labeled_array, num_features = ndimage.label(binarized)
    sizes = ndimage.sum(binarized, labeled_array, range(num_features + 1))

    # threshold for getting rid of small rois: 30 -> 72, 40 -> 63, 50 -> 54
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
            deltaf[i, j] = roi_trace[j] - f0[i]

    # distance matrix
    corrcoef_matrix = np.corrcoef(deltaf)
    # clip negative correlations to 0 before converting to distance
    # this means anti-correlated ROIs are treated as maximally distant (distance=1)
    # rather than super-distant (distance up to 2), which gives HDBSCAN better density structure
    corrcoef_clipped = np.clip(corrcoef_matrix, 0, 1)
    distance = 1 - corrcoef_clipped
    distance = (distance + distance.T) / 2
    distance = np.clip(distance, 0, None)  # remove floating point negatives
    np.fill_diagonal(distance, 0)  # force exact zero after all operations

    # distance distribution plots to visualise cluster structure
    condensed = distance[np.triu_indices(distance.shape[0], k=1)]

    fig_dist, axes_dist = plt.subplots(1, 3, figsize=(18, 5))
    fig_dist.suptitle(f"Distance structure — {filename}", fontsize=11)

    # histogram of all pairwise distances
    axes_dist[0].hist(condensed, bins=50, color="steelblue", edgecolor="none")
    axes_dist[0].set_xlabel("Distance (1 - correlation)")
    axes_dist[0].set_ylabel("Count")
    axes_dist[0].set_title("Pairwise distance distribution")

    # sorted distance plot — gaps suggest natural cluster boundaries
    axes_dist[1].plot(np.sort(condensed), color="steelblue", linewidth=1)
    axes_dist[1].set_xlabel("Pair rank")
    axes_dist[1].set_ylabel("Distance")
    axes_dist[1].set_title("Sorted pairwise distances (steps/elbows = cluster boundaries)")

    # heatmap of the raw distance matrix, ordered by ROI index
    im = axes_dist[2].imshow(distance, cmap="viridis", vmin=0, vmax=1)
    axes_dist[2].set_title("Distance matrix (ROI order)")
    axes_dist[2].set_xlabel("ROI")
    axes_dist[2].set_ylabel("ROI")
    plt.colorbar(im, ax=axes_dist[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_dist.savefig(os.path.join(output, f"{filename}_distance_structure.svg"))
    plt.close(fig_dist)

    # hdbscan clustering on precomputed distance matrix
    min_cluster_size = 3
    max_cluster_size = max(2, num_features // 2)
    db = hdbscan_pkg.HDBSCAN(min_cluster_size=min_cluster_size, max_cluster_size=max_cluster_size, metric='precomputed')
    raw_labels = db.fit_predict(distance)
    probabilities = db.probabilities_

    # noise ROIs (label == -1) get their own individual cluster labels
    n_real = len(set(raw_labels) - {-1})
    cluster_labels = raw_labels.copy()
    next_noise_label = n_real
    for i in range(len(cluster_labels)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = next_noise_label
            next_noise_label += 1
    n_clusters = next_noise_label

    # print cluster summary
    noise_count = int((raw_labels == -1).sum())
    print(f"\nHDBSCAN results for {filename}:")
    print(f"  min_cluster_size={min_cluster_size}, max_cluster_size={max_cluster_size}")
    print(f"  clusters found: {n_real}  |  noise ROIs: {noise_count}  |  total: {num_features}")
    for c in range(n_real):
        members = np.where(raw_labels == c)[0]
        avg_prob = probabilities[members].mean()
        print(f"  cluster {c+1}: {len(members)} ROIs, mean membership probability={avg_prob:.3f}")
    print()

    # reorder for correlation matrix display
    # extract upper triangle manually to avoid squareform validation issues
    n = distance.shape[0]
    condensed = distance[np.triu_indices(n, k=1)]
    linkage_matrix = linkage(condensed, method='average')
    cluster_order = leaves_list(linkage_matrix)
    corrcoef_reordered = np.corrcoef(deltaf[cluster_order])

    output = output_dir

    # color coding: noise ROIs get gray
    set1 = [plt.cm.Set1(i) for i in range(9)]
    set2 = [plt.cm.Set2(i) for i in range(8)]
    set3 = [plt.cm.Set3(i) for i in range(12)]
    all_colors = set1 + set2 + set3  # 29 total
    colors = []
    for i in range(n_clusters):
        if i < n_real:
            colors.append(all_colors[i % len(all_colors)])
        else:
            colors.append((0.7, 0.7, 0.7, 1.0))  # gray for noise

    # membership probability plot
    fig_prob, ax_prob = plt.subplots(figsize=(max(8, num_features * 0.2), 4))
    roi_indices = np.arange(num_features)
    bar_colors = [colors[cluster_labels[i]][:3] for i in roi_indices]
    ax_prob.bar(roi_indices, probabilities, color=bar_colors, edgecolor="none")
    ax_prob.set_xlabel("ROI index")
    ax_prob.set_ylabel("Membership probability")
    ax_prob.set_title(f"HDBSCAN membership probabilities (gray=noise) — {filename}")
    ax_prob.set_xticks(roi_indices)
    ax_prob.set_xticklabels([str(i + 1) for i in roi_indices], fontsize=6, rotation=90)
    ax_prob.axhline(0.5, color="tomato", linewidth=1, linestyle="--", label="p=0.5")
    ax_prob.legend(fontsize=8)
    plt.tight_layout()
    fig_prob.savefig(os.path.join(output, f"{filename}_hdbscan_probabilities.svg"))
    plt.close(fig_prob)

    # average traces of each cluster
    plt.figure(1, figsize=(20, 15))
    groups = [[] for _ in range(n_clusters)]
    for cluster_idx in range(n_clusters):
        roi_indices = np.where(cluster_labels == cluster_idx)[0]
        plt.subplot(n_clusters, 1, cluster_idx + 1)
        for roi_idx in roi_indices:
            groups[cluster_idx].append(roi_idx + 1)
        if len(roi_indices) > 0:
            avg_trace = np.mean(deltaf[roi_indices], axis=0)
            plt.plot(avg_trace, color=colors[cluster_idx], linewidth=1.5)
            label = f'noise\n(n={len(roi_indices)})' if cluster_idx >= n_real else f'C{cluster_idx+1}\n(n={len(roi_indices)})'
            plt.ylabel(label, fontsize=7, rotation=0, labelpad=35)
        plt.tick_params(labelbottom=(cluster_idx == n_clusters - 1))
    plt.figure(1).savefig(os.path.join(output, f"{filename}_clusters_over_time.svg"))

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
    plt.title(f"HDBSCAN clusters (gray=noise)")

    plt.subplot(1, 2, 2)
    plt.imshow(corrcoef_reordered, cmap='Spectral', vmin=-1, vmax=1)
    ticks = np.arange(num_features)
    tick_labels = [str(cluster_order[i] + 1) for i in range(num_features)]
    plt.xticks(ticks, tick_labels, rotation=90, fontsize=6)
    plt.yticks(ticks, tick_labels, fontsize=6)
    plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
    plt.figure(2).savefig(os.path.join(output, f"{filename}_groups_and_matrix.svg"))

    # dendrogram: leaves coloured by hdbscan cluster
    leaf_colors = {roi_idx: colors[cluster_labels[roi_idx]] for roi_idx in range(num_features)}

    fig_dend, ax_dend = plt.subplots(figsize=(max(12, num_features * 0.25), 6))
    dendrogram(
        linkage_matrix,
        ax=ax_dend,
        labels=[str(i + 1) for i in range(num_features)],
        leaf_font_size=6,
        link_color_func=lambda _: "#aaaaaa",
        leaf_label_func=lambda i: str(i + 1),
    )
    for lbl in ax_dend.get_xticklabels():
        roi_idx = int(lbl.get_text()) - 1
        c = leaf_colors[roi_idx]
        lbl.set_color(c[:3])
    ax_dend.set_title(f"All-ROI dendrogram (coloured by HDBSCAN cluster) — {filename}")
    ax_dend.set_ylabel("Distance")
    plt.tight_layout()
    fig_dend.savefig(os.path.join(output, f"{filename}_dendrogram_all.svg"))
    plt.close(fig_dend)

if __name__ == "__main__":
    print('FINDROIS only')
    subfolders = [f.path for f in os.scandir("readfiles") if f.is_dir()]

    for foldername in subfolders:
        # make subfolder in output folder
        output = os.path.join("outputs", os.path.splitext(foldername)[0])
        os.makedirs(output, exist_ok=True)
        print(f'folder: {output}')

        # run code
        data = np.load(f'{foldername}/data.npy')
        cross_corr = np.load(f'{foldername}/cross_corr.npy')
        findroi(data, cross_corr, filename=foldername, output_dir=output)