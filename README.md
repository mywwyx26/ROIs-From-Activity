github doesn't allow files larger than 25mb so all the other files are in google drive, this is just for the code that i need to keep updating

LIST OF HARDCODED PARTS:
- cross corr is sigma = 1 and w = 3
- findrois threshold for small rois is 50 pixels

Clustering:
- tried kmeans, but could not avoid the hard coding and it wasn't very fitting anyway
- discovered hierarchical, which worked until i redid the binarize (otsu's method to avoid hard coding)
- discovered dbscan which seems like a better option, since it can identify noise, not pushed to git
- while searching for more info on dbscan, i discovered hdbscan, which will hopefully work better since the arguments are automatically computed instead of manually setting a distance