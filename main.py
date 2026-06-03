import os
from readfiles import readfile
from findrois import findroi

folder = "C:\\Users\\megan\\flies\\ROIsFromActivity\\inputs"

tif_files = [
    os.path.join(folder, f)
    for f in sorted(os.listdir(folder))
    if f.endswith(".tif")
]

for filename in tif_files:
    # make subfolder in output folder
    basename = os.path.basename(filename)
    output = os.path.join("outputs", os.path.splitext(basename)[0])
    os.makedirs(output, exist_ok=True)
    print(f'folder created: {output}')

    # run code
    data, cross_corr = readfile(filename, sigma=1, w=3)
    print(f'readfiles done: {basename}')
    findroi(data, cross_corr, filename=basename, output=output)
    print(f'findrois done: {basename}')