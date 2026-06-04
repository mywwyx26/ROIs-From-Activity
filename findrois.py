import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, leaves_list, dendrogram, fcluster
from sklearn.metrics import silhouette_samples
from skimage.filters import threshold_otsu

# main function
def findroi(data, cross_corr, filename, output_dir="outputs"):
    os.makedirs(output_dir, exist_ok=True)
    output = output_dir

    # binarize cross corr image using Otsu's method
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
    fig.savefig(os.path.join(output, f"{filename}_threshold_comparison.svg"))
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

    # distance matrix — full range, no clipping of negative correlations
    corrcoef_matrix = np.corrcoef(deltaf)
    distance = 1 - corrcoef_matrix
    distance = (distance + distance.T) / 2
    distance = np.clip(distance, 0, None)
    np.fill_diagonal(distance, 0)

    # distance distribution plots
    condensed = distance[np.triu_indices(distance.shape[0], k=1)]

    fig_dist, axes_dist = plt.subplots(1, 3, figsize=(18, 5))
    fig_dist.suptitle(f"Distance structure — {filename}", fontsize=11)

    axes_dist[0].hist(condensed, bins=50, color="steelblue", edgecolor="none")
    axes_dist[0].set_xlabel("Distance (1 - correlation)")
    axes_dist[0].set_ylabel("Count")
    axes_dist[0].set_title("Pairwise distance distribution")

    axes_dist[1].plot(np.sort(condensed), color="steelblue", linewidth=1)
    axes_dist[1].set_xlabel("Pair rank")
    axes_dist[1].set_ylabel("Distance")
    axes_dist[1].set_title("Sorted pairwise distances (steps/elbows = cluster boundaries)")

    im = axes_dist[2].imshow(distance, cmap="viridis", vmin=0, vmax=2)
    axes_dist[2].set_title("Distance matrix (ROI order)")
    axes_dist[2].set_xlabel("ROI")
    axes_dist[2].set_ylabel("ROI")
    plt.colorbar(im, ax=axes_dist[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig_dist.savefig(os.path.join(output, f"{filename}_distance_structure.svg"))
    plt.close(fig_dist)

    # hierarchical clustering
    linkage_matrix = linkage(condensed, method='average')
    cluster_order = leaves_list(linkage_matrix)
    corrcoef_reordered = np.corrcoef(deltaf[cluster_order])

    # find best k using mean silhouette score
    max_k = min(20, num_features - 1)
    best_k, best_score, best_raw_labels = 2, -1, None
    all_scores = {}
    for k in range(2, max_k + 1):
        labels = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_samples(distance, labels, metric='precomputed').mean()
        all_scores[k] = score
        if score > best_score:
            best_score, best_k, best_raw_labels = score, k, labels

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

    print(f"\nClustering results for {filename}:")
    print(f"  hierarchical (average linkage) + silhouette noise detection")
    print(f"  best k={best_k} (mean silhouette={best_score:.4f})")
    print(f"  clusters: {n_real}  |  noise ROIs: {noise_count}  |  total: {num_features}")
    for c in range(n_real):
        members = np.where(cluster_labels == c)[0]
        avg_sil = per_roi_scores[members].mean() if len(members) > 0 else 0
        print(f"  cluster {c+1}: {len(members)} ROIs, mean silhouette={avg_sil:.3f}")
    print()

    # color coding: noise ROIs get gray
    set1 = [plt.cm.Set1(i) for i in range(9)]
    set2 = [plt.cm.Set2(i) for i in range(8)]
    set3 = [plt.cm.Set3(i) for i in range(12)]
    all_colors = set1 + set2 + set3
    colors = []
    for i in range(n_clusters):
        if i < n_real:
            colors.append(all_colors[i % len(all_colors)])
        else:
            colors.append((0.7, 0.7, 0.7, 1.0))  # gray for noise

    # silhouette plots
    fig_sil, axes_sil = plt.subplots(1, 2, figsize=(14, 4))
    fig_sil.suptitle(f"Silhouette analysis — {filename}", fontsize=11)

    ks = sorted(all_scores.keys())
    axes_sil[0].plot(ks, [all_scores[k] for k in ks], marker='o', color="steelblue", linewidth=1.5)
    axes_sil[0].axvline(best_k, color="tomato", linewidth=1.2, linestyle="--",
                        label=f"best k={best_k} ({best_score:.4f})")
    axes_sil[0].set_xlabel("k")
    axes_sil[0].set_ylabel("Mean silhouette score")
    axes_sil[0].set_title("Silhouette score vs k")
    axes_sil[0].set_xticks(ks)
    axes_sil[0].legend(fontsize=8)

    roi_idx_arr = np.arange(num_features)
    bar_colors = [colors[cluster_labels[i]][:3] for i in roi_idx_arr]
    axes_sil[1].bar(roi_idx_arr, per_roi_scores, color=bar_colors, edgecolor="none")
    axes_sil[1].axhline(noise_threshold, color="tomato", linewidth=1.2, linestyle="--",
                        label=f"noise threshold ({noise_threshold})")
    axes_sil[1].set_xlabel("ROI index")
    axes_sil[1].set_ylabel("Silhouette score")
    axes_sil[1].set_title("Per-ROI silhouette (gray=noise)")
    axes_sil[1].set_xticks(roi_idx_arr)
    axes_sil[1].set_xticklabels([str(i + 1) for i in roi_idx_arr], fontsize=6, rotation=90)
    axes_sil[1].legend(fontsize=8)

    plt.tight_layout()
    fig_sil.savefig(os.path.join(output, f"{filename}_silhouette.svg"))
    plt.close(fig_sil)

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
            label = f'noise\n(n=1)' if cluster_idx >= n_real else f'C{cluster_idx+1}\n(n={len(roi_indices)})'
            plt.ylabel(label, fontsize=7, rotation=0, labelpad=35)
        plt.tick_params(labelbottom=(cluster_idx == n_clusters - 1))
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

    # dendrogram: leaves coloured by cluster
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
    cut_height = linkage_matrix[-(best_k - 1), 2]
    ax_dend.axhline(cut_height, color="tomato", linewidth=1.2, linestyle="--",
                    label=f"k={best_k} cut ({cut_height:.3f})")
    ax_dend.set_title(f"All-ROI dendrogram (coloured by cluster, gray=noise) — {filename}")
    ax_dend.set_ylabel("Distance")
    ax_dend.legend(fontsize=8)
    plt.tight_layout()
    fig_dend.savefig(os.path.join(output, f"{filename}_dendrogram_all.svg"))
    plt.close(fig_dend)


if __name__ == "__main__":
    print('FINDROIS only')
    subfolders = [f.path for f in os.scandir("readfiles") if f.is_dir()]

    for foldername in subfolders:
        print(foldername)
        output = os.path.join("outputs", os.path.basename(foldername))
        os.makedirs(output, exist_ok=True)
        print(f'folder: {output}')

        data = np.load(f'{foldername}/data.npy')
        cross_corr = np.load(f'{foldername}/cross_corr.npy')
        findroi(data, cross_corr, filename=os.path.basename(foldername), output_dir=output)