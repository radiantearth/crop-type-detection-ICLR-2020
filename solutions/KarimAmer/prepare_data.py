import datetime
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from utils import load_file

def create_dataset(bands_arr, radius):
    imgs = np.zeros((4688,13,13,radius*2,radius*2), dtype = np.float32)
    areas = np.zeros((4688,), dtype = np.int)
    gts = np.zeros((4688,), dtype = np.int)
    field_masks = np.zeros((4688,1,radius*2,radius*2), dtype = np.float32)
    fields_arr = []
    ifx = 0
    
    for tile in range(4):
        #load field id and label matrices
        fids = f'{args.data_path}/0{tile}/{tile}_field_id.tif'
        labs = f'{args.data_path}/0{tile}/{tile}_label.tif'
        field_id = load_file(fids)
        labels = load_file(labs)
        #create a grid of pixel positions to use for cropping
        grid = np.indices(field_id.shape)
        
        for field in np.unique(field_id):
            if field == 0:
              continue
            fields_arr.append(field)
            #find pixels belong to current field id
            area_mask = field_id == field
            #extract ground-truth class
            area_gt = np.unique(labels[area_mask])[0] 
            #calculate the median pixel position to crop around it
            idxx = np.where(area_mask)
            momentx = np.median(idxx[0]).astype(np.int)
            momenty = np.median(idxx[1]).astype(np.int)
            #create crop
            patch = bands_arr[tile,:,:,max(0, momentx-radius): momentx+radius, max(0, momenty-radius): momenty+radius]
            #pad crops in tiles borders with zeros
            imgs[ifx, :, :, :patch.shape[-2], :patch.shape[-1]] = patch
            #create crop's field mask (1s for pixels belong to current field id and zeros otherwise)
            field_patch = area_mask[max(0, momentx-radius): momentx+radius, max(0, momenty-radius): momenty+radius]
            #pad crop's field mask in tiles borders with zeros
            field_masks[ifx, 0, :patch.shape[-2], :patch.shape[-1]] = field_patch
            #make sure the crop's field mask is not empty
            if field_patch.sum() == 0:
                print(ifx, momentx-radius, momentx+radius, momenty-radius, momenty+radius)
            #calculate field area
            areas[ifx] = area_mask.sum()
            gts[ifx] = area_gt - 1
            ifx += 1
    return imgs, areas, gts, field_masks, fields_arr

parser = argparse.ArgumentParser(description='Data preperation')
parser.add_argument('-dp','--data_path', help='path to data folder', default='data', type=str)
parser.add_argument('-sp','--save_path', help='save path for generated crops', default='.', type=str)
args = parser.parse_args()

# List of dates that an observation from Sentinel-2 is provided in the training dataset
dates = [datetime.datetime(2019, 6, 6, 8, 10, 7),
         datetime.datetime(2019, 7, 1, 8, 10, 4),
         datetime.datetime(2019, 7, 6, 8, 10, 8),
         datetime.datetime(2019, 7, 11, 8, 10, 4),
         datetime.datetime(2019, 7, 21, 8, 10, 4),
         datetime.datetime(2019, 8, 5, 8, 10, 7),
         datetime.datetime(2019, 8, 15, 8, 10, 6),
         datetime.datetime(2019, 8, 25, 8, 10, 4),
         datetime.datetime(2019, 9, 9, 8, 9, 58),
         datetime.datetime(2019, 9, 19, 8, 9, 59),
         datetime.datetime(2019, 9, 24, 8, 9, 59),
         datetime.datetime(2019, 10, 4, 8, 10),
         datetime.datetime(2019, 11, 3, 8, 10)]

bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']

bands_arr = np.zeros((4,13,13,3035,2016), dtype = np.float32)
#read all images
for tile in range(4):
  for idx, d in enumerate(dates): # 2) For each date
    d = ''.join(str(d.date()).split('-')) # Nice date string
    t = '0'+str(tile)
    for ibx, b in enumerate(bands): # 3) For each band
      # Load im
      im = load_file(f"{args.data_path}/{t}/{d}/{t[1]}_{b}_{d}.tif").astype(np.float32)
      bands_arr[tile,idx,ibx] = im

#create crops of 32X32 around each field id center
imgs, areas, gts, field_masks, fields_arr = create_dataset(bands_arr, 16)

#save data
np.save(os.path.join(args.save_path, 'imgs_13_ch_rad_16_medianxy'), imgs)
np.save(os.path.join(args.save_path, 'areas'), areas)
np.save(os.path.join(args.save_path, 'gts'), gts)
np.save(os.path.join(args.save_path, 'field_masks_medianxy'), field_masks)
np.save(os.path.join(args.save_path, 'fields_arr'), np.array(fields_arr))

del bands_arr
