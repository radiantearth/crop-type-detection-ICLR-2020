import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import pandas as pd
#import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.model_selection import StratifiedShuffleSplit
import datetime
import argparse
from dataset import ICLRDataset
from model import ConvGRUNet
from utils import test, train_model_snapshot, load_file

parser = argparse.ArgumentParser(description='Ensemble training script')
parser.add_argument('-dp','--data_path', help='path to data folder', default='data', type=str)
parser.add_argument('-cp','--crops_path', help='path to generated crops', default='.', type=str)
parser.add_argument('-ssp','--sample_sub_path', help='path to sample submission file including the file name', default='SampleSubmission.csv', type=str)
parser.add_argument('-sp','--save_path', help='save path for output submission file', default='.', type=str)
parser.add_argument('-ma','--mixup_augmentation', help='True for enabling training the model with mixup augmentation', default=False, type=bool)
args = parser.parse_args()

if args.mixup_augmentation:
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

    #load all tiles
    bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']

    tile = np.zeros((4,13,13,3035,2016), dtype = np.float32)

    for itx in range(4):
        for idx, d in enumerate(dates): # 2) For each date
            d = ''.join(str(d.date()).split('-')) # Nice date string
            t = '0'+str(itx)
            for ibx, b in enumerate(bands): # 3) For each band
                # Load im
                im = load_file(f"{args.data_path}/{t}/{d}/{t[1]}_{b}_{d}.tif").astype(np.float32)
                #if (ibx == 1):
                #  print(im.max(), im.min())
                tile[itx,idx,ibx] = im
else:
    tile = None


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

#set all seeds
torch.manual_seed(0)
np.random.seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#read preprocessed data
imgs = np.load(os.path.join(args.crops_path, 'imgs_13_ch_rad_16_medianxy.npy'))
areas = np.load(os.path.join(args.crops_path, 'areas.npy'))
gts = np.load(os.path.join(args.crops_path, 'gts.npy'))
fields_arr = np.load(os.path.join(args.crops_path, 'fields_arr.npy'))
field_masks = np.load(os.path.join(args.crops_path, 'field_masks_medianxy.npy'))

#generate vegitation indecies for training and testing data
ndvi = (imgs[:,:,7:8,:,:] - imgs[:,:,3:4,:,:]) / (imgs[:,:,7:8,:,:] + imgs[:,:,3:4,:,:] + 1e-6)
ndwi_green = (imgs[:,:,2:3,:,:] - imgs[:,:,7:8,:,:]) / (imgs[:,:,2:3,:,:] + imgs[:,:,7:8,:,:] + 1e-6)
ndwi_blue = (imgs[:,:,1:2,:,:] - imgs[:,:,7:8,:,:]) / (imgs[:,:,1:2,:,:] + imgs[:,:,7:8,:,:] + 1e-6)

if args.mixup_augmentation:
    #generate vegitation indecies for all tiles
    ndvi_tile = (tile[:,:,7:8,:,:] - tile[:,:,3:4,:,:]) / (tile[:,:,7:8,:,:] + tile[:,:,3:4,:,:] + 1e-6)
    ndwi_green_tile = (tile[:,:,2:3,:,:] - tile[:,:,7:8,:,:]) / (tile[:,:,2:3,:,:] + tile[:,:,7:8,:,:] + 1e-6)
    ndwi_blue_tile = (tile[:,:,1:2,:,:] - tile[:,:,7:8,:,:]) / (tile[:,:,1:2,:,:] + tile[:,:,7:8,:,:] + 1e-6)

#apply sqrt to lower skewness
imgs = np.sqrt(imgs)
if args.mixup_augmentation:
    tile = np.sqrt(tile)

imgs = np.concatenate([imgs, ndvi, ndwi_green, ndwi_blue], axis = 2)
if args.mixup_augmentation:
    tile = np.concatenate([tile, ndvi_tile, ndwi_green_tile, ndwi_blue_tile], axis = 2)

#data standardization
for c in range(imgs.shape[2]):
    mean = imgs[:, :, c].mean()
    std = imgs[:, :, c].std()
    imgs[:, :, c] = (imgs[:, :, c] - mean) / std
    if args.mixup_augmentation:
        tile[:, :, c] = (tile[:, :, c] - mean) / std

sss = StratifiedShuffleSplit(n_splits=10, test_size=0.15, random_state=0)

print(imgs.shape)

models_arr = []
fold = 0
for train_index, val_index in sss.split(areas[gts > -1], gts[gts > -1]):
    print(fold)
    fold += 1
    image_datasets = {'train': ICLRDataset(tile, imgs, areas, gts, field_masks, 'train', train_index, args.mixup_augmentation),
		      'val': ICLRDataset(None, imgs, areas, gts, field_masks, 'val', val_index),
		      'test': ICLRDataset(None, imgs, areas, gts, field_masks, 'test', None)}

    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=16, shuffle=True, num_workers=16),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=16, shuffle=False, num_workers=16)}

    model_ft = ConvGRUNet(imgs.shape[2]-1)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    dataset_sizes = {x:len(image_datasets[x]) for x in ['train', 'val']}
    
    #train a model on this data split using snapshot ensemble
    model_ft_arr, _, _ = train_model_snapshot(model_ft, criterion, 0.008, dataloaders, dataset_sizes, device,
                           num_cycles=6, num_epochs_per_cycle=10)
    models_arr.extend(model_ft_arr)

#predict on test set using avg of all snapshots of all splits 
test_fields_arr = fields_arr[gts == -1]
test_loader = torch.utils.data.DataLoader(image_datasets['test'], batch_size=16,shuffle=False, num_workers=4)
res = test(models_arr, test_loader, device)

#make a submission
sub = pd.read_csv(args.sample_sub_path)
sub['Field_ID'] = test_fields_arr.tolist()
for i in range(res.shape[1]):
    sub['Crop_ID_%d'%(i+1)] = res[:,i].tolist()

sub.to_csv(os.path.join(args.save_path, 'sub.csv'), index = False)
