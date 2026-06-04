import os
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, leaves_list, fcluster, dendrogram
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

# helper function: find best k using silhouette score on precomputed distance matrix
def best_k_from_linkage(lm, dist_matrix, min_k=2, max_k=20):
    n = dist_matrix.shape[0]
    max_k = min(max_k, n - 1)  # can't have more clusters than samples - 1

    best_k, best_score, best_labels = min_k, -1, None
    all_scores = {}  # k -> silhouette score

    for k in range(min_k, max_k + 1):
        labels = fcluster(lm, t=k, criterion='maxclust') - 1
        if len(np.unique(labels)) < 2:
            continue
        score = silhouette_score(dist_matrix, labels, metric='precomputed')
        all_scores[k] = score
        if score > best_score:
            best_score, best_k, best_labels = score, k, labels

    if best_labels is None:  # fallback
        best_labels = fcluster(lm, t=min_k, criterion='maxclust') - 1
        best_k = min_k
        all_scores[min_k] = -1

    # pick first k where score is greater than the previous k's score
    ks = sorted(all_scores.keys())
    chosen_k, chosen_labels = ks[0], fcluster(lm, t=ks[0], criterion='maxclust') - 1
    for i in range(1, len(ks)):
        if all_scores[ks[i]] > all_scores[ks[i - 1]]:
            chosen_k = ks[i]
            chosen_labels = fcluster(lm, t=chosen_k, criterion='maxclust') - 1
            break

    return chosen_labels, chosen_k, all_scores

# main function
def findroi(data, cross_corr, filename, output_dir="outputs"):
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

    # histogram with both thresholds marked
    axes[1, 0].hist(cross_corr.ravel(), bins=100, color="steelblue", edgecolor="none")
    axes[1, 0].axvline(meanstd_thresh, color="tomato",    linewidth=1.5, label=f"mean+stdev ({meanstd_thresh:.3f})")
    axes[1, 0].axvline(otsu_thresh,    color="limegreen", linewidth=1.5, label=f"Otsu ({otsu_thresh:.3f})")
    axes[1, 0].set_title("Pixel intensity histogram")
    axes[1, 0].set_xlabel("Cross-correlation value")
    axes[1, 0].set_ylabel("Count")
    axes[1, 0].legend(fontsize=8)

    # difference image: pixels that disagree between the two methods
    diff = binarized.astype(int) - binarized_meanstd.astype(int)
    diff_rgb = np.zeros((*diff.shape, 3))
    diff_rgb[diff ==  1] = [0, 1, 0]   # green: Otsu keeps, mean+stdev drops
    diff_rgb[diff == -1] = [1, 0, 0]   # red:   mean+stdev keeps, Otsu drops
    axes[1, 1].imshow(diff_rgb)
    axes[1, 1].set_title("Difference (green=Otsu only, red=mean+stdev only)")
    axes[1, 1].axis("off")

    # pixel counts summary
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

    # plotting things and save them in output folder
    output = output_dir

    # single cut: find best k across all ROIs using silhouette score
    cluster_labels, n_clusters, silhouette_scores = best_k_from_linkage(linkage_matrix, distance, min_k=2, max_k=20)

    # print silhouette scores for all k
    print(f"\nSilhouette scores for {filename}:")
    for k, score in sorted(silhouette_scores.items()):
        marker = " <-- best" if k == n_clusters else ""
        print(f"  k={k}: {score:.4f}{marker}")
    print()

    # plot silhouette scores
    fig_sil, ax_sil = plt.subplots(figsize=(8, 4))
    ks = sorted(silhouette_scores.keys())
    scores = [silhouette_scores[k] for k in ks]
    ax_sil.plot(ks, scores, marker='o', color="steelblue", linewidth=1.5)
    ax_sil.axvline(n_clusters, color="tomato", linewidth=1.2, linestyle="--",
                   label=f"best k={n_clusters} ({silhouette_scores[n_clusters]:.4f})")
    ax_sil.set_xlabel("Number of clusters (k)")
    ax_sil.set_ylabel("Silhouette score")
    ax_sil.set_title(f"Silhouette scores — {filename}")
    ax_sil.set_xticks(ks)
    ax_sil.legend(fontsize=8)
    plt.tight_layout()
    fig_sil.savefig(os.path.join(output, f"{filename}_silhouette_scores.svg"))
    plt.close(fig_sil)

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

    # dendrogram: all ROIs, leaves coloured by final cluster
    leaf_colors = {roi_idx: colors[cluster_labels[roi_idx]] for roi_idx in range(num_features)}

    def leaf_color_fn(roi_idx):
        c = leaf_colors[roi_idx]
        return f"rgb({int(c[0]*255)},{int(c[1]*255)},{int(c[2]*255)})"

    fig_dend, ax_dend = plt.subplots(figsize=(max(12, num_features * 0.25), 6))
    dendrogram(
        linkage_matrix,
        ax=ax_dend,
        labels=[str(i + 1) for i in range(num_features)],
        leaf_font_size=6,
        link_color_func=lambda _: "#aaaaaa",
        leaf_label_func=lambda i: str(i + 1),
    )
    # colour the x-tick labels by cluster
    for lbl in ax_dend.get_xticklabels():
        roi_idx = int(lbl.get_text()) - 1
        c = leaf_colors[roi_idx]
        lbl.set_color(c[:3])
    # mark the best-k cut height
    cut_height = linkage_matrix[-(n_clusters - 1), 2]
    ax_dend.axhline(cut_height, color="tomato", linewidth=1.2, linestyle="--",
                    label=f"k={n_clusters} cut ({cut_height:.3f})")
    ax_dend.set_title(f"All-ROI dendrogram — {filename}")
    ax_dend.set_ylabel("Distance")
    ax_dend.legend(fontsize=8)
    plt.tight_layout()
    fig_dend.savefig(os.path.join(output, f"{filename}_dendrogram_all.svg"))
    plt.close(fig_dend)