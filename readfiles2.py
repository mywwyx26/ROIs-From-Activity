# the imports have squigglies but it works if i do >python readfiles.py
import numpy as np
import matplotlib.pyplot as plt
import tifffile
import napari
import time
from scipy.ndimage import gaussian_filter

# each array is just a bunch of numbers in the dimensions of (time, x, y)
data = tifffile.imread("TSeries-12232025-1359-444_Cycle00001_frames2883to4482_reg.tif")

# gausian blur on the whole video
def gaussian_blur_video(array, sigma=1): # time: 3.6s for sigma=1, 6.8s for sigma=3, 10s for sigma=5
    # initialize empty array
    blurred_array = np.zeros(array.shape)

    # apply gaussian filter to each frame
    for i in range(array.shape[0]):
        blurred_array[i, :, :] = gaussian_filter(array[i, :, :], sigma=sigma)
    return blurred_array

# remove the bg, return the new array with bg = 0 and a list of pixels that are not bg, helper for cross corr image
# bg is the p percentile of darkest pixels
def remove_bg(array, p=30): # time: ~1.2s for 30%
    copy = array.copy()
    pixels = []
    mean = np.mean(copy, axis=0)
    mean_th = np.percentile(np.ravel(mean), p)
    num_frames, xmax, ymax = copy.shape

    for x in range(xmax):
        for y in range(ymax):
            if mean[x, y] < mean_th:
                copy[:, x, y] = 0
            else:
                pixels.append((x, y))
    return copy, pixels

# cross corr images function from https://labrigger.com/blog/2013/06/13/local-cross-corr-images/
# very slow so try to find or make a better one
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

blurred_data1 = gaussian_blur_video(data, sigma=1)
blurred_data3 = gaussian_blur_video(data, sigma=3)
blurred_data5 = gaussian_blur_video(data, sigma=5)

'''
viewer = napari.Viewer()
viewer.add_image(data, name='original')
viewer.add_image(blurred_data1, name='gaussian blur sigma=1')
viewer.add_image(blurred_data3, name='gaussian blur sigma=3')
viewer.add_image(blurred_data5, name='gaussian blur sigma=5')
napari.run()
'''

copy0, pixels0 = remove_bg(data)
copy1, pixels1 = remove_bg(blurred_data1)
copy3, pixels3 = remove_bg(blurred_data3)
copy5, pixels5 = remove_bg(blurred_data5)

plt.subplot(4,2,1)
plt.imshow(np.mean(copy0, axis=0), cmap='rainbow')
plt.subplot(4,2,3)
plt.imshow(np.mean(copy1, axis=0), cmap='rainbow')
plt.subplot(4,2,5)
plt.imshow(np.mean(copy3, axis=0), cmap='rainbow')
plt.subplot(4,2,7)
plt.imshow(np.mean(copy5, axis=0), cmap='rainbow')
plt.show()