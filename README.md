github doesn't allow files larger than 25mb so all the other files are saved locally, this is just for the code that i need to keep updating

how to use:
- option 1: run main.py only
- option 2: run readfiles.py, wait for it to finish, then run findrois.py

file descriptions:
- main.py: runs readfiles and findrois on all files in the inputs folder, saves .svg files to output folder
- readfiles.py: do cross corr of the tif file, return np arrays of video and cross corr image
    - when executed directly, takes all inputs from input folder and saves .npy files to readfiles folder
- findrois.py: binarize the cross corr image, filter out rois smaller than 50px, deltaf/f for normalization, get correlation coefficient matrix, do clustering (see below), graph everything
    - when executed directly, takes all inputs from readfiles folder and saves .svg files to output folder

hardcoded parts: (trust that these all worked best)
- cross corr is sigma = 1 and w = (x+y)/2 * 0.005
- binarize threshold is 80% (keeps brightest 20%)
- threshold for small rois is x * y * 0.0002
- min clusters = 2
- max clusters = ROIs/2, capped at 20
- noise threshold = 0.0
- there's probably more

clustering:
- tried kmeans, but could not avoid the hard coding and it wasn't very fitting anyway
- discovered hierarchical, which worked until i redid the binarize (otsu's method to avoid hard coding)
- discovered dbscan which seems like a better option, since it can identify noise, not pushed to git
- while searching for more info on dbscan, i discovered hdbscan, which will hopefully work better since the arguments are automatically computed instead of manually setting a distance
- i am going back to hierarchical clustering bc hdbscan did not work very well either
- which works fine since otsu's method was bad, so we're back to mean + stdev binarize
