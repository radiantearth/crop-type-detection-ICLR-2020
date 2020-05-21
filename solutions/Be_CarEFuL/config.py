import os
from os import listdir 
from os.path import join
RANDOM_STATE = 2020
absolut_path="/workspace/Zindi/ICLR_2"
raw_data_path=join(absolut_path,"data/raw_data")
proc_data_path=join(absolut_path,"data/proc_data")
other_data_path=join(absolut_path,"data/other_data")
sub_path=join(absolut_path,"data/sub")
oof_train_path=join(absolut_path,"outputs/oof/train")
oof_test_path=join(absolut_path,"outputs/oof/test")
expirment_path=join(absolut_path,"outputs/experiments.csv")
tem_data_path=join(absolut_path,"data/tmp_data")
bands = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12', 'CLD']
swap_prob=0.005
