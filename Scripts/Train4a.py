# Imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle

import os
import cv2
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

# Basic Functions
def cdf(df):
    df0 = df
    rm = []
    for i in df['filename']:
        if(type(cv2.imread(i))==type(None)):
            print(i)
            rm.append(i)
            df0 = df0[df0['filename']!=i]
    print(df.shape)
    print('Removed ',len(rm),' files')
    return reindex(df0)  

def reindex(df):
    df['Index'] = np.arange(len(df))
    return df.set_index('Index')

def htg(im):
    im = cv2.equalizeHist(im)
    #img = gaussian_filter(im,sigma=25)
    #im = im*(img>40)
    return im

def imP3(im):
    im1,im2,im3 = im[:,:,0],im[:,:,1],im[:,:,2]
    im[:,:,0],im[:,:,1],im[:,:,2] = htg(im1), htg(im2), htg(im3)
    im = cv2.resize(im,(103,128))
    return im

# DataLoader
class DataLoader(Dataset):  
    
    def __init__(self, df, trans):
        super(DataLoader,self)        
        self.transforms = trans
        self.df = df
        self.path = '../'
        
    def __getitem__(self,i):        
        self.labels = Variable(torch.FloatTensor(self.df[self.df.columns[1:]].loc[i].values.astype(float))).cuda()
        self.image = Variable(torch.FloatTensor(imP3(cv2.imread(self.path+self.df['filename'][i])))).cuda()
                 
        return self.image.view(1,3,103,128),self.labels
    
    def __len__(self):
        return len(self.df)

def dl(df):
    return DataLoader(df,transforms.Compose([transforms.Normalize, transforms.RandomCrop, transforms.RandomSizedCrop]))
    
# Define Model Class

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
        
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=isTrained)
        kernelCount = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(kernelCount, classCount))
        self.dropout = nn.Dropout2d(p=0.3, inplace=False)

    def forward(self, x):
        
        x = self.net.conv1(x)
        x = self.net.bn1(x)
        x = self.net.relu(x)
        x = self.net.maxpool(x)

        x = self.net.layer1(x)
        x = self.dropout(x)
        x = self.net.layer2(x)
        x = self.dropout(x)
        x = self.net.layer3(x)
        x = self.dropout(x)
        x = self.net.layer4(x)
        x = self.net.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.net.fc(x)

        return F.relu(x)

    
# Define Loss Function

def dla(yp,yt):
    return F.mse_loss(yp,yt)*len(yp)
    
def sumc(l):
    u = 0*l[0]
    for i in l:
        u+=i
    return u

def genw(df_train):
    s = max([len(df_train[df_train[i]==1]) for i in df_train.columns[1:]])
    W = (np.array([len(df_train[df_train[i]==1]) for i in df_train.columns[1:]])/s)**(-1)
    W = W/np.max(W)
    wd = dict(zip(np.arange(len(df_train.columns[1:])),W))
    return wd
    
def genc(n):
    z = []
    for i in range(n):
        u = torch.zeros(n).cuda()
        u[i]=1
        z.append(u)
    return z
    
class Loss1(nn.Module):
    
    def __init__(self,alf,lam,n=4,sig=1.5,a=2):
      
        super(Loss1, self).__init__()
        self.C = genc(n)
        self.alf = alf
        self.lam = lam
        self.sig = sig
        self.a = a
        self.w0 = genw(df)
        
    def forward(self,yp,yt):
        
        #La = torch.sum(self.a*torch.exp( ((yp - yt)/self.sig)**2 ))
        La = dla(yp,self.alf*yt)
        Lt = torch.log(1+sumc([torch.exp( La - dla(yp,self.alf*i)) for i in self.C ]))
        L = Lt + self.lam*La
        L = L*self.w0[torch.argmax(yt).item()]
        if(L<1.5):
            L = L*1.8
        return 1.5*L

# Define Train Function

def train(Model,n_epochs):
    
    elosses_mse = []
    elosses_wgcac = []
    optimizer = torch.optim.Adam(Model.parameters(), lr=0.01,weight_decay=0.001) 
    #optimizer = optim.SGD(Model.parameters(), lr=0.05, momentum=0.9)

    criterion = Loss1(1,2)
    mse = nn.MSELoss()
    
    for i in range(n_epochs):
    
        print()
        print('Starting epoch: ', i+1)
        print()
        eloss_wgcac = 0
        eloss_mse = 0
        
        for j in range(15):
            
            bloss_wgcac = 0
            bloss_mse = 0

            #print('Batch Number: ',u)
            
            optimizer.zero_grad()
            
            ux = np.random.randint(1,DL.__len__()-1,40)

            for u in ux:

                #print('Batch Index: ', batch_index)
                image,Labels = DL.__getitem__(u)
                output = Model(image)[0]                
                bloss_wgcac += criterion(output, Labels) 
                bloss_mse += mse(output, Labels).item()
            
            avg_bloss_wgcac = bloss_wgcac/40
            avg_bloss_mse = bloss_mse/40
               
            avg_bloss_wgcac.backward()
            optimizer.step()
            
            eloss_wgcac += avg_bloss_wgcac.item()
            eloss_mse += avg_bloss_mse
            
        elosses_mse.append(eloss_mse/15)
        elosses_wgcac.append(eloss_wgcac/15)
        print()
        print('Epoch MSE: ', eloss_mse/15) 
        print('Epoch Wgcac Loss ',eloss_wgcac/15)
        print()
        
    return elosses_wgcac, elosses_mse
    

# Load Model
if(int(input('First Attempt: '))):
    n3 = ResNet18(2,False)
    n3.load_state_dict(torch.load('../Models/Model3.pth.tar'))
    n3.net.fc = nn.Sequential(nn.Linear(512, 4))
    n3 = n3.cuda()
else:
    n3 = ResNet18(4,False)
    n3.load_state_dict(torch.load(input('Enter model path ')))
    n3 = n3.cuda()

# Load Metadata
df = pd.read_csv('../Mets/Train4a.csv')
df = df[df.columns[1:]]
#df = cdf(df)
DL = dl(df)

qq = DL.__getitem__(1);
#print('Path available', qq[0].shape, qq[1].shape, n3(qq[0])[0].shape)
model_title = input('Enter Model Title: ')
LW, LMS = train(n3,int(input('Enter number of epochs: ')))
dff = pd.DataFrame()
dff['WGCAC'] = LW
dff['MSE'] = LMS
dff.to_csv('../Losses/'+model_title + 'Losses.csv')
torch.save(n3.state_dict(),'../Models/' + model_title + '.pth.tar')
print('Saved at: ', '../Models/' + model_title + '.pth.tar')
