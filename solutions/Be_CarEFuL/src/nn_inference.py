import warnings

warnings.filterwarnings("ignore")

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
from scipy.stats import spearmanr, rankdata
from os.path import join

import sys
import time 
sys.path.append("../")
from helper.utils import *
from config import *
from tqdm import tqdm
random_seed_cpu_gpu(RANDOM_STATE)


Print(" "*25+"nn_inference")

from sklearn.metrics import log_loss
def metric(x, y):
    return log_loss(x, y)
prediction_names=["Crop_ID_"+str(i) for i in range(1,8)]


class data_set(Dataset):
    def __init__(
        self,
        data=None,
        is_train=True
    ):
        self.data = data
        self.is_train = is_train

    def __getitem__(self, index):
        row = self.data.iloc[index]
        x=row["features"]
        fid=row["fid"]
        if self.is_train:
            target=row["label"]
            return x,target,fid
        else:
            return x,fid

    def __len__(self):
        return len(self.data)
def get_train_val_loader(train,fold, train_batch_size=32, val_batch_size=64):
    val_data = train[train.fold==fold]
    train_data = train[train.fold!=fold]
    
    train_data_set = data_set(data=train_data)
    val_data_set = data_set(data=val_data)

    train_loader = DataLoader(
        train_data_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_data_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )

    return train_loader, val_loader


def get_test_loader(test,test_batch_size=64):
    test_data_set = data_set(data=test,is_train=False)
    test_loader = DataLoader(
        test_data_set,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return test_loader

class classifier_model (nn.Module):
    
    def __init__(self,
             
                 n_input_chanells,
                n_latent_chanells,
                kernl_size,
                padding,n_class=7
                ) :
        
        super(classifier_model, self).__init__()

        self.lin1=nn.Linear(n_input_chanells,n_latent_chanells,bias=True)
        self.lin2=nn.Linear(n_latent_chanells,n_latent_chanells*2,bias=True)
        self.lin3=nn.Linear(n_latent_chanells*2,n_latent_chanells*4,bias=True)
        
        self.logit=nn.Linear(n_latent_chanells*4,n_class,bias=True)
        
        
    def forward(self,x):
        Conv1d_1=self.lin1(x)
        Conv1d_2=self.lin2(Conv1d_1)
        Conv1d_3=self.lin3(Conv1d_2)
        logit=self.logit(Conv1d_3)

        

        return logit
def load_model(fold,model_name) :
    path=join(other_data_path,"nn","fold_"+str(int(fold)),model_name)
    model=classifier_model(n_input_chanells, n_latent_chanells, kernl_size, padding)
    model.to(device)
    epoch=10000
    try : 
        model.load_state_dict(torch.load(path ,map_location=device))
    except :
        print("user_ last model ")
        model.load_state_dict(torch.load(path ,map_location=device))

    return model


def predict(model_name,fold,data_loader,with_label=True):
    print("#"*10+" "+str(fold)+"#"*10)

    model=load_model(fold,model_name)
    predictions=[]
    targets=[]
    fid=[]
    model.eval()
    with torch.no_grad():
        for step, (inputs) in enumerate((data_loader)):
            if  with_label : 
                x,labels,fids=inputs
                targets.append(labels.cpu())
            else  :
                x,fids=inputs
                
            x=x.to(device).float()
            logits= model(x)
            predictions.append(F.softmax((logits).cpu()))
            fid.append(fids.cpu())
        predictions = torch.cat(predictions, 0).detach().numpy()
        fid = torch.cat(fid, 0).detach().numpy()
        
        if  with_label : 
            targets = torch.cat(targets, 0).detach().numpy()
            return fid ,predictions,targets  
        else : 
            return fid,predictions
        

train=read_pickle(join(proc_data_path,"train_nn.p"))
test=read_pickle(join(proc_data_path,"test_nn.p"))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Print("Device to use by  torch is : "+ device.type)

n_input_chanells=train.features.iloc[0].shape[-1]
n_latent_chanells=512
kernl_size=5
padding=3

model_name="nn_model.p"
n_epochs = 10
train_batch_size, val_batch_size = 64, 128
num_workers=8
patience=100
init_epoch=-1
Print("predict validation ")

validations=[]
for fold in np.sort(train.fold.unique()):
    _,val_data_loader = get_train_val_loader(
        train,fold, train_batch_size, val_batch_size
    )
    fid,predictions,targets=predict(model_name,fold,val_data_loader,with_label=True)
    validation=pd.DataFrame()
    validation["fid"]=fid
    validation["labels"]=targets    
    validation=fill_predictions_df(validation,predictions,prediction_names)
    val_id=validation.groupby("fid").mean()
    print("loss: ",metric(val_id["labels"],val_id[prediction_names]))
    print("loss: ",metric(validation["labels"],validation[prediction_names]))
    validations.append(val_id)
validations=pd.concat(validations)   
print("loss: ",metric(validations["labels"],validations[prediction_names]))
score=metric(validations["labels"],validations[prediction_names])
Print("validation score :"+str(score))

Print("predict test ")
test_preds=[]
for fold in np.sort(train.fold.unique()):
    test_data_loader=get_test_loader(test)
    fid,predictions=predict(model_name,fold,test_data_loader,with_label=False)
    test_pred=pd.DataFrame()
    test_pred["fid"]=fid
    test_pred=fill_predictions_df(test_pred,predictions,prediction_names)
    test_pred["fold"]=fold
    test_preds.append(test_pred)
test_preds=pd.concat(test_preds)  
test_preds.groupby(["fid"]).mean().drop("fold",1).to_csv(join(sub_path,"nn_model"),index=True)
Print("Save sub to "+join(sub_path,"nn_model"))
