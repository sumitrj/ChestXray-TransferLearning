# Imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import os
import cv2
import pydicom
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms


# Preprocssing Functions
def htg(im):
    im = cv2.equalizeHist(im)
    img = gaussian_filter(im,sigma=25)
    im = im*(img>50)
    return im

def imP3(im):
    im1,im2,im3 = im[:,:,0],im[:,:,1],im[:,:,2]
    im[:,:,0],im[:,:,1],im[:,:,2] = htg(im1), htg(im2), htg(im3)
    im = cv2.resize(im,(103,128))
    return im
    return im

# DataLoader class definition
class DataLoader(Dataset):  
    
    def __init__(self, df, trans):
        
        super(DataLoader,self)
        self.df = df
        self.path = '../../COVID-19/IMAGES (copy)/IMAGES (copy)/(1)'
        self.transforms = trans
        
    def __getitem__(self,i):
        
        self.labels = Variable(torch.FloatTensor(self.df[self.df.columns[1:]].loc[i].values.astype(float))).cuda()
        self.image = Variable(torch.FloatTensor(imP3(cv2.imread(self.path+self.df['filename'][i])))).cuda()
                 
        return self.image.view(1,3,103,128),self.labels
    
    def __len__(self):
        return len(self.df)

def dl(df):
    return DataLoader(df,transforms.Compose([transforms.Normalize, transforms.RandomCrop, transforms.RandomSizedCrop]))


# Define Network Class
class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
        
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=isTrained)
        kernelCount = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.net(x)
        return F.relu(x)

# Define Loss Function

"""
We have udeveloped and proposed novel uncertainty based loss functions which has resulted in fast training and accuracte predictions.
We won't disclose the functions
Similar to definition of the Network class, different mathematical constructs can be used in the myLoss class definition
"""
    
class myLoss(nn.Module):
    
    def __init__(self,alf,lam,n=15):
      
        super(myLoss, self).__init__()
        
        # Define your variables here
        
    def forward(self,yp,yt):
        
        # Enter all the computational instructions here
        
        #Dummy line
        L = torch.sum(torch.abs(yp-yt))

        return L

# Define Training Function

def train(n_epochs,Model):
    
    epoch_mse_losses = []
    epoch_myLoss_losses = []
    criterion = myLoss()
    mse = nn.MSELoss()
    
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.01,weight_decay=0.001)
    
    for i in range(n_epochs):
    
        print()
        print('Starting epoch: ', i+1)
        print()
        epoch_mse_loss = 0
        epoch_myLoss_loss = 0
        
        for u,(j,k)in enumerate(TrainLoader3):
            
            batch_mse = 0
            batch_myLoss = 0
            optimizer.zero_grad()
            
            for batch_index in range(5):

                Image = j[batch_index].view(1,3,103,128)
                Labels = k[batch_index]

                output = Model(Image)[0]                
                
                myLoss_loss = criterion(output, Labels) 
                mse_loss = mse(output, Labels).item()
                batch_myLoss+=myLoss_loss
                batch_mse+=mse_loss
            
            mean_batch_myLoss = batch_myLoss/5
            mean_batch_mse = batch_mse/5
       
            mean_batch_myLoss.backward()
            optimizer.step()
            
        epoch_mse_loss = mean_batch_mse/u
        epoch_myLoss_loss = (mean_batch_myLoss/u).item
        
        epoch_mse_losses.append(epoch_mse_loss)
        epoch_myLoss_losses.append(epoch_myLoss_loss)
        
        print('Epoch MSE Loss: ', epoch_mse_loss) 
        print('Epoch myLoss Loss: ', epoch_myLoss_loss)
        
    return epoch_myLoss_losses, epoch_mse_losses
    
    
# Load Metadata       
df = pd.read_csv('../Mets/Train2.csv')
df = df[df.columns[1:]]
DL = dl(df)
TrainLoader3 = torch.utils.data.DataLoader(dataset = DL, batch_size = 7, shuffle = True)

# Load Model:
if(int(input('Load Trained? '))):
    Net = ResNet18(4,False)
    Net.load_state_dict(torch.load(input('Enter Base Model Path: ')))
else:
    Net = ResNet18(2,False)
    Net.load_state_dict(torch.load('../Models/Model1.pth.tar'))
    Net.net.fc = nn.Sequential(nn.Linear(512, 4))                  
Net1 = Net.cuda()

# Get Model title, number of epochs and train:
model_title = input('Enter Model title: ')
MyLoss, MSE = train(Net1,int(input('Enter Epochs: ')))
df = pd.DataFrame()
df['MyLoss Loss'] = MyLoss
df['MSE Loss'] = MSE
df.to_csv('../Losses/' + model_title + 'Losses.csv')
torch.save(Net1.state_dict(), '../Models/' + model_title + '.pth.tar')
print('Saved at: ', '../Models/' + model_title + '.pth.tar')
