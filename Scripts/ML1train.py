# Imports
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import pickle
from scipy.ndimage import gaussian_filter
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# General Functions

# Check if filenames of non-existing images are present in the dataframe
def cdf(df):
    df0 = df
    rm = []
    for i in df['filename']:
        if(type(cv2.imread(i))==type(None)):
            #print(i)
            rm.append(i)
            df0 = df0[df0['filename']!=i]
    #print(df.shape)
    #print('Removed ',len(rm),' files')
    return reindex(df0)  

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

# Generate features for ResNet18 model and DataLoader
def genx(Model,DL):
    
    Model.net.fc = Identity()
    X = []
    for i in range(DL.__len__()):
        X.append(Model(DL.__getitem__(i)[0]).detach().cpu().numpy())
        #print(i)
    X = np.array(X)
    X2 = X.reshape(X.shape[0],X.shape[-1])
    return X2

# Image Procesing
def htg(im):
    im = cv2.equalizeHist(im)
    img = gaussian_filter(im,sigma=25)
    im = im*(img>30)
    return im

def imP3(im):
    
    im1,im2,im3 = im[:,:,0],im[:,:,1],im[:,:,2]
    im[:,:,0],im[:,:,1],im[:,:,2] = htg(im1), htg(im2), htg(im3)
    im = cv2.resize(im,(103,128))
    return im

# DataLoader Class Definition
class DataLoader(Dataset):  
    
    def __init__(self, df, path, trans):
        
        super(DataLoader,self)        
        self.transforms = trans
        self.df = df
        self.path = path
        
    def __getitem__(self,i):
        
        self.labels = Variable(torch.FloatTensor(self.df[self.df.columns[1:]].loc[i].values.astype(float))).cuda()
        self.image = Variable(torch.FloatTensor(imP3(cv2.imread(self.path+self.df['filename'][i])))).cuda()
                 
        return self.image.view(1,3,103,128),self.labels
    
    def __len__(self):
        return len(self.df)
    
def dl(df,path):
    return DataLoader(df,path,transforms.Compose([transforms.Normalize, transforms.RandomCrop, transforms.RandomSizedCrop]))

def reindex(df):
    df['Index'] = np.arange(len(df))
    return df.set_index('Index')

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

# Training function for Random Forest Classifier
def MLtrain(X,y):
    clf = RandomForestClassifier(max_depth=4000, random_state=0)
    clf.fit(X,y)
    return clf
    
# Load Metadata
df = pd.read_csv('../Mets/Train4a.csv')
df = df[df.columns[1:]]
df = reindex(pd.concat([df[df[i]==1].iloc[:] for i in df.columns]))
DL = dl(df,'')

# Load Network
net = ResNet18(4,False).cuda()
net.load_state_dict(torch.load('../Models/Model42(1).pth.tar'))

# Train Model
X = genx(net,DL)
clf = MLtrain(X,df['COVID-19'].values)
# Save Model
file = '../Models/ML1.sav'
pickle.dump(clf, open(file, 'wb'))
print('COVID-19 Model Saved at ',file)
clf1 = MLtrain(X,df['Normal'].values)
file = '../Models/ML2.sav'
pickle.dump(clf1, open(file, 'wb'))
print('Normal Model Saved at ',file)
