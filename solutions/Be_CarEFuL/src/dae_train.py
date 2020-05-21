import warnings

warnings.filterwarnings("ignore")
import sys
import time 
sys.path.append("../")
from helper.utils import pd,random_seed_cpu_gpu,Print,read_pickle
from config import *
from tqdm import tqdm
from os.path import join

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


class Ae_data(Dataset):
    def __init__(
        self,
        data,
        ids,
        swap_prob=swap_prob
    ):
        self.data=data
        self.ids=ids
        self.swap_prob=swap_prob

    def __getitem__(self, index):
        
        x=self.data[index]
        p=np.random.rand()<=self.swap_prob
        if p :
            swap_index=np.random.randint(self.__len__())
            return x ,self.data[swap_index]
        else : 
            return x,x
     
    def __len__(self):
        return len(self.data)   

def train_func(
    model,
    criterion,
    optimizer,
    train_data_loader,
    model_name,
    n_epochs,
    init_epoch=0,
    path=""
    
):
    train_losses = []
    
    os.makedirs(path, exist_ok=True)
    for epoch in range(init_epoch,n_epochs+1):
        iterator_train = tqdm(train_data_loader)
        epoch_train_losses = []
        tr_loss=0
        model.train()
        for step, (x,y) in enumerate(iterator_train):
            x =   x.to(device).float()
            y=   y.to(device).float()
            
            outputs=model(x)
            loss = torch.sqrt(criterion(y, outputs))
            loss.backward()
            optimizer.step()
            model.zero_grad()
            epoch_train_losses.append(loss.item())
            tr_loss+=loss.item()
            iterator_train.set_postfix(epoch=epoch,loss=(loss.item()), loss_total=(tr_loss/((step+1))) )
            
        train_losses.append(np.mean(epoch_train_losses))
        if epoch%10==0:
            torch.save(model.encoder.state_dict(),join(path,"encoder_{}".format(epoch)))
            torch.save(model.decoder.state_dict(), join(path,"decoder_{}".format(epoch)))
            

    return train_losses


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
        

Print(" "*25+" Train denoising Autoencoder ")
Print("")
Print("1-Load the seq data ")
data=read_pickle(join(proc_data_path,"seq_data.p"))


train_batch_size=32
num_workers=8
model_name="final"
init_epoch=-1


Print("2- Create torch data loader ")

train_set = Ae_data(data=list(data.values()),ids=list(data.keys()))

train_loader = DataLoader(
    train_set,
    batch_size=train_batch_size,
    shuffle=True,
    num_workers=num_workers,
    drop_last=True,
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Print("Device to use by  torch is : "+ device.type)


Print("Create DAE model")
encoder=encoder_model(14)
decoderl=decoder_model(14,14)
ae=AE_model(encoder,decoderl)
ae.to(device)

optimizer = optim.Adam(ae.parameters(), lr=1e-4, eps=1e-8)
criterion = nn.MSELoss()


path =join(other_data_path,"ae",model_name)
if init_epoch > -1 :
    ae.encoder.load_state_dict(torch.load(join(path,"encoder_{}".format(epoch)) ,map_location=device))
    ae.decoder.load_state_dict(torch.load(join(path,"decoder_{}".format(epoch))  ,map_location=device))
    
n_epochs=2500
Print("Start training ")
start_time=time.time()
train_func(
    ae,
    criterion,
    optimizer,
    train_loader,
    model_name,
    n_epochs,
    init_epoch+1,
    path=path
)
end_time=start_time-time.time()