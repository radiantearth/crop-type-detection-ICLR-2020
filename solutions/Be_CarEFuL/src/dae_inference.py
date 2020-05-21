import warnings

warnings.filterwarnings("ignore")
import sys
import time 
sys.path.append("../")
from helper.utils import pd,random_seed_cpu_gpu,Print,read_pickle,save_pickle
from config import *
from tqdm import tqdm
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
random_seed_cpu_gpu(RANDOM_STATE)


from sklearn.metrics import mean_squared_error
def metric(x,y):
    return np.sqrt(mean_squared_error(x.reshape(-1,14),y.reshape(-1,14)))


class Ae_data(Dataset):
    def __init__(
        self,
        data,
        ids
    ):
        self.data=data
        self.ids=ids

    def __getitem__(self, index):
        x=self.data[index]
        return x
     
    def __len__(self):
        return len(self.data)  
    
    
class encoder_model(nn.Module):
    def __init__(
        self,
        n_input
       
    ):
        super(encoder_model, self).__init__()
        self.lstm_1 = nn.LSTM(n_input,14, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(14*2,14, num_layers=1, batch_first=True, bidirectional=True)
        self.latent = nn.LSTM(14*2,14,batch_first=True)

        

    def forward(self, x):
        lstm_1=self.lstm_1(x)[0]
        lstm_2=self.lstm_2(lstm_1)[0]
        latent=self.latent(lstm_2)[0]
       
        return  latent
    
class decoder_model(nn.Module):
    def __init__(
        self,
        n_input,
        n_output
       
    ):
        super(decoder_model, self).__init__()
        self.lstm_1 = nn.LSTM(n_input,14, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_2 = nn.LSTM(14*2,14, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm_3 = nn.LSTM(14*2,14 ,batch_first=True)
        self.output_layer=nn.Linear(14,n_output)
        

    def forward(self, x):
        lstm_1=self.lstm_1(x)[0]
        lstm_2=self.lstm_2(lstm_1)[0]
        lstm_3=self.lstm_3(lstm_2)[0]
        output_layer=self.output_layer(lstm_3)
        return  output_layer
class AE_model(nn.Module):
    def __init__(self,encoder,decoder ):
        super(AE_model, self).__init__()
        self.encoder=encoder
        self.decoder=decoder
    def forward(self, x):
        encoder=self.encoder(x)
        decoder=self.decoder(encoder)
        return  decoder
    
    
train_batch_size=32
num_workers=6

Print(" "*25+"Inference denoising Autoencoder")
Print("")
Print("1-Load the seq data ")
data=read_pickle(join(proc_data_path,"seq_data.p"))


Print("2- Create torch data loader ")
train_set = Ae_data(data=list(data.values()),ids=list(data.keys()))
train_loader = DataLoader(
    train_set,
    batch_size=train_batch_size,
    shuffle=False,
    num_workers=num_workers,
    drop_last=False,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Print("Device to use by  torch is : "+ device.type)


model_name="final"
path =join(other_data_path,"ae",model_name)
encoder=encoder_model(14)
decoderl=decoder_model(14,14)
ae=AE_model(encoder,decoderl)
ae=ae.to(device)
ae.encoder.load_state_dict(torch.load(join(path,"encoder.p") ,map_location=device))
ae.decoder.load_state_dict(torch.load(join(path,"decoder.p" ) ,map_location=device))

ae.eval()
predictions=[]
true_labels=[]
features=[]
for step, x in enumerate(train_loader):
    x =   x.to(device).float()
    feature=ae.encoder(x)
    outputs=ae(x)
    predictions.append(outputs.cpu())
    true_labels.append(x.cpu())
    features.append(feature.cpu())
predictions = torch.cat(predictions, 0).detach().numpy()
true_labels = torch.cat(true_labels, 0).detach().numpy()
features = torch.cat(features, 0).detach().numpy()

score=metric(true_labels,predictions)
Print("AE rmse : "+str(score))

features_df=pd.DataFrame()
features_df["id"]=list(data.keys())
features_df["fid"]=features_df.id.apply(lambda x:x.split("_")[0]).astype(int)
features_df["row_loc"]=features_df.id.apply(lambda x:x.split("_")[1]).astype(int)
features_df["col_loc"]=features_df.id.apply(lambda x:x.split("_")[2]).astype(int)
features_df["features"]=list(features)
features_df.drop(["id"],1,inplace=True)
Print("Save learned features to "+"Ae_features_{}.p".format(str(round(score,3))) )
save_pickle(features_df,join(proc_data_path,"Ae_features_{}.p".format(str(round(score,3)))))

Print("Process learned features for NN model")
features_df=read_pickle(join(proc_data_path,"Ae_features_{}.p".format(str(round(score,3)))))
target=pd.read_pickle(join(proc_data_path,"target.p"))
folds=pd.read_csv(join(other_data_path,"cross_validation_index_fold_5.csv"))
features_df["features"]=features_df["features"].apply(lambda x:x.reshape(-1))
features_df=features_df.groupby("fid").features.apply(lambda x:np.mean(x)).rename("features").reset_index()
features_df=features_df.merge(target,on="fid",how="left")
features_df=features_df.merge(folds,on="fid",how="left")
train_nn=features_df[~features_df.label.isna()]
test_nn=features_df[features_df.label.isna()]
train_nn.to_pickle(join(proc_data_path,"train_nn.p"))
test_nn.to_pickle(join(proc_data_path,"test_nn.p"))
del train_nn,test_nn,features_df


Print("Process learned features for catboost model")
data = pd.read_pickle(join(proc_data_path,"data.p"))
_dl_features_=read_pickle(join(proc_data_path,"Ae_features_0.042.p"))
__features__=np.array(_dl_features_.features.apply(list).tolist())
features_num=__features__.shape[-1]


dl_features=[]
for i,date in enumerate(np.sort(data.date.unique())):
    __dl_features__=_dl_features_[["fid","row_loc","col_loc"]].copy()
    __dl_features__["date"]=date
    df=pd.DataFrame(data=__features__[:,i,:],columns=["Dl_feat_"+str(k) for k in range(features_num)])
    __dl_features__=pd.concat([__dl_features__,df],1)
    dl_features.append(__dl_features__)
dl_features=pd.concat(dl_features)
dl_features=dl_features.astype({"fid":int,
                   "row_loc":int,
                   "col_loc":int})



data["area"]=data.groupby(["fid"]).fid.transform("count")/13
data=data.merge(dl_features,on=["fid","row_loc","col_loc","date"],how="left")
data=data.merge(target,how="left",on=["fid"])
data.drop(bands,1,inplace=True)




test=data[data.train==0]
data=data[data.train==1]
data.label=data.label.astype(int)


ratio_per_target={k:v for k,v in zip(target.label.value_counts(True).index.astype(int),target.label.value_counts(True).values)}
min_area=5
data_2_len_coff=1

data_1=data[data.area<min_area]
data_1=data_1.groupby(["fid","date"]).mean().reset_index()
data_2=data[data.area>=min_area]
test=test.groupby(["fid","date"]).mean().reset_index()

len_data_1=len(data_1)
len_data_2=len_data_1*data_2_len_coff


def sample_data(data_label,ratio):
    supposed_len=round((len_data_2/13)*ratio)+1
    data_label_unique=data_label.drop_duplicates(["fid","row_loc","col_loc","tile"])[["fid","row_loc","col_loc","tile"]]
    data_label_unique=data_label_unique.sample(frac=1)
    new_counts=dict(((data_label_unique.fid.value_counts(True)*supposed_len).round()).astype(int).apply(lambda x: 1 if x==0 else x ))
    old_counts=dict((data_label_unique.fid.value_counts()))
    data_label_unique["sepration"]=np.nan
    for fid in tqdm(new_counts.keys()) : 
        new_count=new_counts[fid]
        old_count=old_counts[fid]
        coff=int(old_count/new_count)
        iter_=int(old_count/coff)+1
        data_label_unique.loc[data_label_unique.fid==fid,"sepration"]=np.array([ [i]*coff for i in range(iter_)]).reshape(-1)[:old_count]
    data_label=data_label.merge(data_label_unique,on=["fid","row_loc","col_loc","tile"],how="left")
    return data_label
        
sampled_data_2=[]
for i in range(7):
    sampled_data_2.append(sample_data(data_2[data_2.label==i],ratio_per_target[i]))
sampled_data_2=pd.concat(sampled_data_2)            

sampled_data_2=sampled_data_2.groupby(["fid","date","sepration"]).mean().reset_index()
sampled_data_2.drop(["sepration"],1,inplace=True)

data=pd.concat([data_1,sampled_data_2,test])
del data["label"]

data=pd.pivot_table(data,index=["fid","row_loc","col_loc","train","tile","area"],columns=["date"])
data.columns=[col[0]+"_"+col[1] for col in data.columns.tolist()]
data.reset_index(inplace=True)

train=data[data.train==1]
test=data[data.train==0]
train=train.merge(target,how="left",on=["fid"])
train=train.merge(folds,on=["fid"],how="left")

train.drop(["row_loc","col_loc","train","tile"],1,inplace=True)
test.drop(["row_loc","col_loc","train","tile"],1,inplace=True)


train.to_pickle(join(proc_data_path,"train_catboost.p"))
test.to_pickle(join(proc_data_path,"test_catboost.p"))