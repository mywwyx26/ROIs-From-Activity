'''
READFILES.PY
    This file takes a calcium imaging recording and computes the cross correlation between each pixel.

Inputs:
    filename (str): .tif file
    sigma (int, default = 1): standard deviation of the gaussian bell curve, higher value = more blurred
    w (int ,default = 3): window of local neighborhood, at 3 this is a 7x7 pixel area
   
Outputs:
    data (3D ndarray): the .tif file as an array
    cross_corr (2D ndarray): the cross correlation image
'''

import os
import numpy as np
import tifffile
from scipy import ndimage

def readfile(filename, sigma = 1, w = 3):
    # each array is just a bunch of numbers in the dimensions of (time, x, y)
    data = tifffile.imread(filename)

    # gausian blur on the whole video
    def gaussian_blur_video(array, sigma=1): # time: 3.6s for sigma=1, 6.8s for sigma=3, 10s for sigma=5
        # initialize empty array
        blurred_array = np.zeros(array.shape)

        # apply gaussian filter to each frame
        for i in range(array.shape[0]):
            blurred_array[i, :, :] = ndimage.gaussian_filter(array[i, :, :], sigma=sigma)
        return blurred_array

    # cross corr images function from https://labrigger.com/blog/2013/06/13/local-cross-corr-images/
    # very slow (w=1 takes 60s, w=2 takes 87s, w=3 takes 117s, w=4 takes 163s)
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

    blurred_data = gaussian_blur_video(data, sigma)
    cross_corr = cross_corr_image(blurred_data, w)
    return data, cross_corr

'''
Takes the .tif files from the inputs folder, then saves data.npy and cross_corr.npy to the readfiles
folder, with subfolders for each recording. Allows for findrois.py to be run multiple times without
having to redo the cross correlation each time.
'''
if __name__ == "__main__":
    print('READFILES only')
    input_folder = "C:\\Users\\megan\\flies\\ROIsFromActivity\\inputs"

    tif_files = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if f.endswith(".tif")
    ]

    for filename in tif_files:
        # make subfolder in output folder
        basename = os.path.basename(filename)
        output = os.path.join("readfiles", os.path.splitext(basename)[0])
        os.makedirs(output, exist_ok=True)

        # run code
        data, cross_corr = readfile(filename, sigma=1, w=3)
        np.save(os.path.join(output, "data.npy"), data)
        np.save(os.path.join(output, "cross_corr.npy"), cross_corr)
        print(f'readfiles done: {output}')
