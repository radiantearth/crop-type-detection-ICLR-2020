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
from os.path import join

import sys
import time 
sys.path.append("../")
from helper.utils import *
from config import *
from tqdm import tqdm
random_seed_cpu_gpu(RANDOM_STATE)


train=read_pickle(join(proc_data_path,"train_nn.p"))
test=read_pickle(join(proc_data_path,"test_nn.p"))
prediction_names=["Crop_ID_"+str(i) for i in range(1,8)]


from sklearn.metrics import log_loss


def metric(x, y):
    return log_loss(x, y)


Print(" "*25+ "NN model")
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, path, patience=7, verbose=False, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.save_path = path

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score - self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), self.save_path)
        self.val_loss_min = val_loss
        
class data_set(Dataset):
    def __init__(
        self,
        data=None,
        is_train=False
    ):
        self.data = data
        self.is_train = is_train

    def __getitem__(self, index):
        row = self.data.iloc[index]
        x=row["features"]
        if self.is_train:
            target=row["label"]
            return x,target
        else:
            return x


    def __len__(self):
        return len(self.data)

    
def get_train_val_loader(train,fold, train_batch_size=32, val_batch_size=64):
    val_data = train[train.fold==fold]
    train_data = train[train.fold!=fold]
    train_data_set = data_set(data=train_data,is_train=True)
    val_data_set = data_set(data=val_data,is_train=True)

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
    
def train_func(
    model,
    cross_entropy_loss,
    optimizer,
    train_data_loader,
    val_data_loader,

    n_epochs,
    patience,
    fold,
    path,
    model_name
):
 
    train_cross_entropy_loss=[]
    train_losses=[]
    
    validation_losses=[]
    validation_cross_entropy_loss=[]
    model_path=join(path,model_name)
    
    early_stopping = EarlyStopping(path=model_path, patience=patience, verbose=True)
    for epoch in range(n_epochs):
        iterator_train = tqdm(train_data_loader)
        epoch_train_cross_entropy_loss=[]
        epoch_train_rmse_loss=[]
        epoch_trian_loss=[]
        _train_cross_entropy_loss_=0
        _train_rmse_loss_=0
        _train_loss=0
        model.train()
        for step, (x, labels) in enumerate(iterator_train):
            x, labels = x.to(device).float(),labels.type(torch.LongTensor).to(device)

            
            
            cl_output = model(x)
            
            loss = cross_entropy_loss(cl_output, labels)
            
            loss.backward()
            optimizer.step()
            model.zero_grad()
            
            epoch_trian_loss.append(loss.item())
            
            _train_loss+=loss.item()

            
            iterator_train.set_postfix(epoch=epoch,
                                       cros_entropy=(_train_loss/(step+1))
                                       )
            
        train_losses.append(np.mean(epoch_trian_loss))
        

        model.eval()
        epoch_validation_cross_entropy_loss=[]
        epoch_validation_rmse_loss=[]
        epoch_validation_loss=[]
        validation_predictions = []
        validation_labels = []
        iterator_val = tqdm(val_data_loader)
        outputs=[]
        target=[]
        for step, (x, labels) in enumerate(iterator_val):
            x, labels =  x.to(device).float(),labels.type(torch.LongTensor).to(device)

            cl_output = model(x)
            outputs.append(F.softmax(cl_output).cpu())
            target.append(labels.cpu())

        outputs = torch.cat(outputs, 0).detach().numpy()
        target = torch.cat(target, 0).detach().numpy()
        epoch_validation_loss=metric(target,outputs)
        print("vali",epoch_validation_loss)
            ###########################
        early_stopping(epoch_validation_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    return 

def create_model(path,model_name,init_epoch) :
    model=classifier_model(n_input_chanells, n_latent_chanells, kernl_size, padding)
    model.to(device)
    
    if init_epoch>-1 :
        model.load_state_dict(torch.load(join(path,model_name) ,map_location=device))
    return model



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Print("Device to use by  torch is : "+ device.type)


model_name="nn_model"
n_epochs = 200
train_batch_size, val_batch_size = 64, 128
num_workers=8
patience=100
init_epoch=-1


n_input_chanells=train.features.iloc[0].shape[-1]
n_latent_chanells=512
kernl_size=5
padding=3

Print("Train NN" )


for fold in np.sort(train.fold.unique()):
    path =join(other_data_path,"nn","fold_"+str(int(fold)))
    os.makedirs(path, exist_ok=True)
    print("#" * 50 + " " + str(fold) + " " + "#" * 50)
    model=create_model(path,model_name,init_epoch)
    train_data_loader, val_data_loader = get_train_val_loader(
        train,fold, train_batch_size, val_batch_size
    )
    steps = len(train_data_loader)
    optimizer = optim.Adam(model.parameters(),lr=1e-4, eps=1e-8)

    criterion = nn.CrossEntropyLoss()
    _=train_func(
        model,
        cross_entropy_loss=criterion,
        optimizer=optimizer,
        train_data_loader=train_data_loader,
        val_data_loader=val_data_loader,
        model_name=model_name,
        n_epochs=n_epochs,
        patience=patience,
        fold=fold,
        path=path,
        )
   