# Imports

import numpy as np
import pandas as pd
from sklearn.utils import shuffle

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

# Define DataLoader
    
def imP3(im):
    im = im[200:,:]
    out = cv2.equalizeHist(im)
    fG = cv2.resize(out,(103,128), interpolation = cv2.INTER_AREA)
    imt = np.zeros((103,128,3))
    for i in range(3): imt[:,:,i] = fG
    imt = cv2.rotate(imt, cv2.ROTATE_180)
    return imt

class DataLoader(Dataset):  
    
    def __init__(self, df, trans, train = 1):
        
        super(DataLoader,self)
        self.df = df
        self.path = '../../rsna-pneumonia-detection-challenge/stage_2_train_images/'        
        self.transforms = trans
        self.train = train
        
    def __getitem__(self,i):
        
        self.image = imP3(pydicom.read_file(self.path+self.df['patientId'][i]+'.dcm').pixel_array)
        self.image = Variable(torch.FloatTensor(self.image)).cuda()
        self.labels = Variable(torch.FloatTensor(self.df[self.df.columns[1:]].iloc[i].values.astype(float))).cuda()
        return self.image.view(1,3,103,128),self.labels
        
    def __len__(self):
        return len(self.df)

def dl(df_train):
        return DataLoader(df_train,transforms.Compose([transforms.Normalize, transforms.RandomCrop, transforms.RandomSizedCrop]))

# Define Model class:

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
     
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=True)
        kernelCount = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = torch.sigmoid(self.net(x))
        return x
        
# Define Training Function

def train(Model,n_epochs):
    
    elosses = []
    loss = 0    
    r2 = F.binary_cross_entropy( Net1(DL.__getitem__(0)[0])[0], DL.__getitem__(0)[1] )
   
    for i in range(n_epochs):
        
        print()
        print('Epoch ', i)
        
        for j in range(10):
            optimizer = optim.SGD(Model.parameters(), lr=((r2.item()>0.7)*0.015 + 0.01), momentum=0.9) 
            eloss = 0
            optimizer.zero_grad() 

            ux = np.random.randint(1,17702,30)
            
            for u in ux:
                image,label = DL.__getitem__(u)
                output = Model(image)[0]
                loss = F.binary_cross_entropy(output, label)
                eloss+=loss

            r2 = eloss/30
            r2.backward()
            optimizer.step()
        
        elosses.append(r2.item())
        print('MSE Loss: ',r2.item())
        
    return elosses 

# Load Metadata
df = pd.read_csv('../Mets/Train1.csv')
df = df[df.columns[1:]]
DL = dl(

# Load Model:
if(int(input('Load Trained? '))):
    Net = ResNet18(2,False)
    Net.load_state_dict(torch.load(input('Enter Base Model Path: ')))
else:
    Net = ResNet18(2,True)
Net1 = Net.cuda()

# Get Model title, number of epochs and run
model_title = input('Enter Model title: ')
Losses = np.array(train(Net1,int(input('Enter Epochs: '))))
L = pd.DataFrame()
L['Losses'] = Loss_trends
L.to_csv('../Losses/' + model_title + 'Losses.csv')
torch.save(Net1.state_dict(), '../Models/' + model_title + '.pth.tar')
print('Saved at: ', '../Models/' + model_title + '.pth.tar')
