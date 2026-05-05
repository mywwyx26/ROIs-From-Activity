import os
from readfiles import readfile
from findrois import findroi

folder = "C:\\Users\\megan\\flies\\flies\\inputs"

tif_files = [
    os.path.join(folder, f)
    for f in sorted(os.listdir(folder))
    if f.endswith(".tif")
]

for filename in tif_files:
    data, cross_corr = readfile(filename, sigma=1, w=3)
    findroi(data, cross_corr, filename=os.path.basename(filename))