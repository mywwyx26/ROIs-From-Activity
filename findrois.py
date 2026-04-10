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
plt.imshow(labeled_array, cmap='nipy_spectral')

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
group1 = [9,7,45,40,42,1,24,2,8,19]
plt.subplot(9, 1, 1)
for i in group1:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group2 = [49,14,33,36,44,34,52,5,28,30,39,41]
plt.subplot(9, 1, 2)
for i in group2:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group3 = [18,27,35,17,51,50,26,43]
plt.subplot(9, 1, 3)
for i in group3:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group4 = [54,46,53]
plt.subplot(9, 1, 4)
for i in group4:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group5 = [4,6,10,15,16,38,12,47]
plt.subplot(9, 1, 5)
for i in group5:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group6 = [20,25,29]
plt.subplot(9, 1, 6)
for i in group6:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group7 = [31,32,37]
plt.subplot(9, 1, 7)
for i in group7:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group8 = [11,22,48]
plt.subplot(9, 1, 8)
for i in group8:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

group9 = [3,13,21]
plt.subplot(9, 1, 9)
for i in group9:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

# plot 5: eyeballing which groups might belong together
plt.figure(5)
concat1 = group6+group9
plt.subplot(7, 1, 1)
for i in concat1:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

concat2 = group7+group8
plt.subplot(7, 1, 2)
for i in concat2:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

concat3 = [25,29,21,20]
plt.subplot(7, 1, 3)
for i in concat3:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

concat4 = [49,14,36,44,34,52,33]
plt.subplot(7, 1, 4)
for i in concat4:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

concat4 = [5,30,39,41,28]
plt.subplot(7, 1, 5)
for i in concat4:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

concat4 = [28,33]
plt.subplot(7, 1, 6)
for i in concat4:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

concat4 = [3,48]
plt.subplot(7, 1, 7)
for i in concat4:
    plt.plot(deltaf[i-1], label=f'ROI {i}')
plt.legend()

# plot 6: final groupings, averaged
group1 = [9,7,45,40,42,1,24,2,8,19]
group2 = [49,14,33,36,44,34,52,5,28,30,39,41]
group3 = [18,27,35,17,51,50,26,43]
group4 = [4,6,10,15,16,38,12,47]
group5 = [20,25,29,21]

avg1 = np.mean([deltaf[i-1] for i in group1], axis=0)
avg2 = np.mean([deltaf[i-1] for i in group2], axis=0)
avg3 = np.mean([deltaf[i-1] for i in group3], axis=0)
avg4 = np.mean([deltaf[i-1] for i in group4], axis=0)
avg5 = np.mean([deltaf[i-1] for i in group5], axis=0)

plt.figure(6)
plt.subplot(5, 1, 1)
plt.plot(avg1, label='Group 1')
plt.legend()
plt.subplot(5, 1, 2)
plt.plot(avg2, label='Group 2')
plt.legend()
plt.subplot(5, 1, 3)
plt.plot(avg3, label='Group 3')
plt.legend()
plt.subplot(5, 1, 4)
plt.plot(avg4, label='Group 4')
plt.legend()
plt.subplot(5, 1, 5)
plt.plot(avg5, label='Group 5')
plt.legend()

plt.show()


