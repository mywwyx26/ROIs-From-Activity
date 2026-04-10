# the imports have squigglies but it works if i do >python findrois.py
import time
import numpy as np
import matplotlib.pyplot as plt
import tifffile
from scipy import ndimage
from scipy.cluster.hierarchy import linkage, leaves_list

# load in np arrays
data = tifffile.imread("TSeries-12232025-1359-444_Cycle00001_frames2883to4482_reg.tif")
cross_corr_image1 = np.load("cross_corr_image1.npy")

# binarize cross corr image
mean1, stdev1 = np.mean(cross_corr_image1), np.std(cross_corr_image1)
binarized1 = np.where(cross_corr_image1 > mean1 + stdev1, 1, 0)
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

# attempt cross correlation and reordering
corrcoef_matrix = np.corrcoef(deltaf)
distance = 1 - corrcoef_matrix
linkage_matrix = linkage(distance, method='average')
cluster_order = leaves_list(linkage_matrix)  # new ordering of ROI indices
deltaf_reordered = deltaf[cluster_order]
corrcoef_reordered = np.corrcoef(deltaf_reordered)

# plot 1: mean, stev, cross corr, binarized, for reference
plt.figure(1)
plt.subplot(2, 2, 1)
plt.imshow(np.mean(data, axis=0), cmap='gray')
plt.subplot(2, 2, 2)
plt.imshow(np.std(data, axis=0), cmap='gray')
plt.subplot(2, 2, 3)
plt.imshow(cross_corr_image1, cmap='gray')
plt.subplot(2, 2, 4)
plt.imshow(binarized1, cmap='gray')

# plot 2: labels reordered by size
plt.figure(2)
plt.subplot(2, 2, 1)
plt.imshow(labeled_array, cmap='nipy_spectral')
plt.subplot(2, 2, 2)
plt.imshow(corrcoef_matrix, cmap='Spectral')
plt.subplot(2, 2, 3)

# plot 3: big labeled diagram of cross corr
plt.figure(3)
plt.imshow(corrcoef_reordered, cmap='Spectral', vmin=-1, vmax=1)
ticks = np.arange(num_features)
labels = [str(cluster_order[i] + 1) for i in range(num_features)]  # +1 for 1-based ROI numbers
plt.xticks(ticks, labels, rotation=90, fontsize=6)
plt.yticks(ticks, labels, fontsize=6)
plt.tick_params(axis='x', top=True, bottom=False, labeltop=True, labelbottom=False)
plt.colorbar()

# plot 4: grouping a bunch together MANUALLY RIP
plt.figure(4)
group1 = [18,27,35,17,51,50,26,43]
plt.subplot(3, 1, 1)
for i in group1:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group2 = [4,6,10,15,16,38,12,47]
plt.subplot(3, 1, 2)
for i in group2:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group3 = [9,7,45,40,42,1,24,2,8,19]
plt.subplot(3, 1, 3)
for i in group3:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

plt.show()


