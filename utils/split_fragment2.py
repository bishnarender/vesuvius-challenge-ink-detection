# This file is a modified version of the original:
# https://github.com/mipypf/Vesuvius-Challenge/blob/winner_call/tattaka_ron/input/split_fragment2.py
    
import glob
import os
import sys
import warnings

import numpy as np
import PIL.Image as Image
from tqdm import trange

warnings.simplefilter("ignore")

PREFIX = "input/train"

data_id = 2
ir = Image.open(PREFIX + f"/{data_id}/ir.png")  # (h, w)
mask = Image.open(PREFIX + f"/{data_id}/mask.png")
label = Image.open(PREFIX + f"/{data_id}/inklabels.png")

volume = [
    Image.open(filename)
    for filename in sorted(glob.glob(PREFIX + f"/{data_id}/surface_volume/*.tif"))
]

# .convert("1") => convert to  1-bit pixels, black and white, stored with one pixel per byte. 1-bit pixel has a range of 0-1.
mask_numpy = np.asarray(mask.convert("1"))

s1 = 0 # upto which row the sum is greater than (1/3)rd of whole sum. treat height indices as rows.
s2 = 0 # upto which row (from s1) the sum is greater than (1/3)rd of whole sum. 
pix_sum = mask_numpy.sum()

# mask_numpy.shape[0] => 14830
# pix_sum => 97993501


for i in trange(mask_numpy.shape[0]):
    s1 += 1
    # mask_numpy[:1], mask_numpy[:1].shape => [[False False False ... False False False]], (1, 9506)
    if mask_numpy[:s1].sum() > pix_sum // 3:
        break
        
s2 = s1
# s2 => 6816
#sys.exit(0)

for i in trange(mask_numpy.shape[0] - s1):
    s2 += 1
    if mask_numpy[s1:s2].sum() > pix_sum // 3:
        break

# s2 => 10571
new_data_ids = [2, 4, 5]
splits = [s1, s2, mask_numpy.shape[0]]

width = mask_numpy.shape[1]
NEW_PREFIX = "vesuvius-challenge-ink-detection-5fold/train"
start = 0

for i, data_id in enumerate(new_data_ids):
    os.makedirs(NEW_PREFIX + f"/{data_id}/", exist_ok=True)
    os.makedirs(NEW_PREFIX + f"/{data_id}/surface_volume/", exist_ok=True)
    
    # .crop(box=left, upper, right, lower)) => box is a 4-tuple defining the left, upper, right, and lower pixel coordinate. 
    ir.crop((0, start, width, splits[i])).save(NEW_PREFIX + f"/{data_id}/ir.png")
    mask.crop((0, start, width, splits[i])).save(NEW_PREFIX + f"/{data_id}/mask.png")    
    label.crop((0, start, width, splits[i])).save(NEW_PREFIX + f"/{data_id}/inklabels.png" )
    
    for j, img in enumerate(volume):
        img.crop((0, start, width, splits[i])).save(NEW_PREFIX + f"/{data_id}/surface_volume/" + "{:02}.tif".format(j))
    start = splits[i]
