# the imports have squigglies but it works if i do >python readfiles.py
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import napari
import time
from scipy import ndimage

# each array is just a bunch of numbers in the dimensions of (time, x, y)
data = tifffile.imread("TSeries-12232025-1359-444_Cycle00001_frames2883to4482_reg.tif")

# gausian blur on the whole video
def gaussian_blur_video(array, sigma=1): # time: 3.6s for sigma=1, 6.8s for sigma=3, 10s for sigma=5
    # initialize empty array
    blurred_array = np.zeros(array.shape)

    # apply gaussian filter to each frame
    for i in range(array.shape[0]):
        blurred_array[i, :, :] = ndimage.gaussian_filter(array[i, :, :], sigma=sigma)
    return blurred_array

# cross corr images function from https://labrigger.com/blog/2013/06/13/local-cross-corr-images/
# very slow so try to find or make a better one (w=1 takes 60s, w=2 takes 87s, w=3 takes 117s, w=4 takes 163s)
def cross_corr_image(tc, w=1):
    num_frames, xmax, ymax = tc.shape
    ccimage = np.zeros((xmax, ymax))

    for x in range(w, xmax - w):
        for y in range(w, ymax - w):

            # Center pixel — shape (T,)
            center_tc = tc[:, x, y]
            thing1 = center_tc - center_tc.mean()           # (T,)
            ad_a   = np.dot(thing1, thing1)                 # scalar

            # Neighborhood — shape (T, 2w+1, 2w+1)
            a      = tc[:, x-w:x+w+1, y-w:y+w+1]
            thing2 = a - a.mean(axis=0, keepdims=True)      # subtract per-pixel mean over time
            ad_b   = np.sum(thing2 * thing2, axis=0)        # (2w+1, 2w+1)

            # Cross-correlation
            numerator = np.einsum('t,txy->xy', thing1, thing2)   # (2w+1, 2w+1)
            denom     = np.sqrt(ad_a * ad_b)

            ccs = np.divide(
                numerator, denom,
                out=np.zeros_like(numerator),
                where=denom != 0
            )

            # Remove center pixel and average
            ccs_flat = ccs.ravel()
            center_idx = ccs_flat.size // 2
            ccs_flat = np.delete(ccs_flat, center_idx)
            ccimage[x, y] = ccs_flat.mean()

    return ccimage

# the one that looked best was sigma = 1 and w = 3
blurred_data1 = gaussian_blur_video(data, sigma=1)
cross_corr_image1 = cross_corr_image(blurred_data1, w=3)

# binarize cross corr image
mean1, stdev1 = np.mean(cross_corr_image1), np.std(cross_corr_image1)
binarized1 = np.where(cross_corr_image1 > mean1 + stdev1, 1, 0)
labeled_array, num_features = ndimage.label(binarized1)
sizes = ndimage.sum(binarized1, labeled_array, range(num_features + 1))
mask = sizes < 30 # threshold for getting rid of small rois, 72 features remain
remove_pixel = mask[labeled_array]
labeled_array[remove_pixel] = 0

# update the image again and reorder the labels by size
labeled_array, num_features = ndimage.label(labeled_array)
label_indices = np.arange(1, num_features + 1)
sizes = ndimage.sum(binarized1, labeled_array, label_indices)
sorted_indices = np.argsort(sizes)[::-1]
ranked_labels = label_indices[sorted_indices]
map_array = np.zeros(num_features + 1, dtype=int)
for i, old_label in enumerate(ranked_labels):
    map_array[old_label] = i + 1
sorted_labeled_mask = map_array[labeled_array]

# get median fluorescence per time for each roi
medians = np.zeros((7,data.shape[0]))
for i in range(7):
    for j in range(data.shape[0]):
        medians[i, j] = ndimage.median(data[j], labels=sorted_labeled_mask, index=i+1)

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
plt.imshow(sorted_labeled_mask, cmap='nipy_spectral')

# plot 3: median fluorescence per time for each roi
plt.figure(3)
plt.subplot(7,1,1)
plt.plot(medians[0])
plt.subplot(7,1,2)
plt.plot(medians[1])
plt.subplot(7,1,3)
plt.plot(medians[2])
plt.subplot(7,1,4)
plt.plot(medians[3])
plt.subplot(7,1,5)
plt.plot(medians[4])
plt.subplot(7,1,6)
plt.plot(medians[5])
plt.subplot(7,1,7)
plt.plot(medians[6])

plt.show()


