import pandas as pd
import os
import argparse

parser = argparse.ArgumentParser(description='Data preperation')
parser.add_argument('-ssp','--subs_path', help='path to the folder that has both best_config_sub.csv and simple_config_sub.csv files', default='.', type=str)
parser.add_argument('-sp','--save_path', help='save path for output submission file', default='.', type=str)
args = parser.parse_args()

sub1 = pd.read_csv(os.path.join(args.subs_path, 'best_config_sub.csv'))
sub2 = pd.read_csv(os.path.join(args.subs_path, 'simple_config_sub.csv'))

for i in range(7):
    sub1['Crop_ID_%d'%(i+1)] = 0.5*sub1['Crop_ID_%d'%(i+1)] + 0.5*sub2['Crop_ID_%d'%(i+1)]

sub1.to_csv(os.path.join(args.save_path, 'final_sub.csv'), index = False)
