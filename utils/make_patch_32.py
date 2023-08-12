# This file is a modified version of the original:
# https://github.com/mipypf/Vesuvius-Challenge/blob/winner_call/tattaka_ron/input/make_patch_32.py

import glob
import os
import gc

import numpy as np
import PIL.Image as Image
from tqdm import tqdm, trange

#import warnings
#warnings.simplefilter("ignore")

PREFIX = "input/train"
PATCH_SIZE = 32

data_ids = [2, 1, 3]

# position=0, => for loop and its nested loops such that there will be a single progress bar (no multiple bars).
for data_id in tqdm(data_ids, position=0, leave=True):
#for data_id in data_ids:    

    ir = np.array(Image.open(PREFIX + f"/{data_id}/ir.png"))
    
    # .convert("1") => convert to  1-bit pixels, black and white, stored with one pixel per byte. 1-bit pixel has a range of 0-1.
    mask = np.array(Image.open(PREFIX + f"/{data_id}/mask.png").convert("1"))
    label = np.array(Image.open(PREFIX + f"/{data_id}/inklabels.png"))
    # label.shape => (14830, 9506)
    
    
    volume=[]
    for filename in sorted(glob.glob(PREFIX + f"/{data_id}/surface_volume/*.tif")): # [:32]
            # np.array(Image.open(filename), dtype=np.float32).max() => 65535.0
            # np.array(Image.open(filename), dtype=np.float32).min() => 0.0
            # 65535.0, which is the maximum value for a 16-bit unsigned integer.
            img = Image.open(filename)
            volume.append(np.array(img, dtype=np.float32) / 65535.0)
            _ = gc.collect()
        
    volume = np.stack(volume)    
    # volume.shape, volume.shape => (65, 14830, 9506)     
    
    assert ir.shape[-2:] == mask.shape[-2:] == volume.shape[-2:]
    
    volume_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/surface_volume/"
    ir_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/ir/"
    mask_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/mask/"
    label_dir = f"vesuvius_patches_{PATCH_SIZE}/train/{data_id}/label/"
    
    os.makedirs(volume_dir, exist_ok=True)
    os.makedirs(ir_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)

    h, w = volume.shape[-2:]
    
    # trange(N) can be also used as a convenient shortcut for tqdm(range(N)).
    # leave=True to keeps all traces of the progressbar upon termination of iteration.
    # h // PATCH_SIZE, w // PATCH_SIZE => 463, 297    
    for i in trange(h // PATCH_SIZE, position=0, leave=False):
        for j in trange(w // PATCH_SIZE, position=0, leave=False):
            start_h = i * PATCH_SIZE
            start_w = j * PATCH_SIZE
            
            # ... (Ellipsis) 
            # [..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE] => adjust the two dimension (from last) and leave rest as it is.
            mask_patch = mask[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            
            if not mask_patch.sum():
                continue
            
            volume_patch = volume[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            # volume_patch.shape => (65, 32, 32)
            
            ir_patch = ir[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            
            
            label_patch = label[
                ..., start_h : start_h + PATCH_SIZE, start_w : start_w + PATCH_SIZE
            ]
            # label_patch.shape => (32,32)
            
            #break # my_
            np.save(os.path.join(volume_dir, f"volume_{i}_{j}"), volume_patch)
            np.save(os.path.join(ir_dir, f"ir_{i}_{j}"), ir_patch)
            np.save(os.path.join(label_dir, f"label_{i}_{j}"), label_patch)
            np.save(os.path.join(mask_dir, f"mask_{i}_{j}"), mask_patch)
        #break # my_
    #break # my_