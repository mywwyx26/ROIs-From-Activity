import os
import numpy as np
import tifffile
import cv2

if __name__ == "__main__":
    print('tiff file to mp4')
    import matplotlib.pyplot as plt
    input_folder = "C:\\Users\\megan\\flies\\ROIsFromActivity\\inputs"

    tif_files = [
        os.path.join(input_folder, f)
        for f in sorted(os.listdir(input_folder))
        if f.endswith(".tif")
    ]

    for filename in tif_files:
        data = tifffile.imread(filename)

        # normalize to 0-255 for video
        data_norm = ((data - data.min()) / (data.max() - data.min()) * 255).astype(np.uint8)

        os.makedirs('mp4s', exist_ok=True)

        height, width = data.shape[1], data.shape[2]
        basename = os.path.splitext(os.path.basename(filename))[0]
        out = cv2.VideoWriter(f'mp4s/{basename}.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height), isColor=False)

        for frame in data_norm:
            out.write(frame)

        out.release()
        print(f'{basename} done')