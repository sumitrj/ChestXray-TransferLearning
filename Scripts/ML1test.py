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
    
def reindex(df):
    df['Index'] = np.arange(len(df))
    return df.set_index('Index')

# Calculate various accuracy parameters for the predictions
def test(yt,yp,th=0.5):
       
        tp,tn,fp,fn = 0,0,0,0
        
        u = []
        
        for i in range(len(yt)): 
            if(yt[i]):
                if(yp[i]>th):
                    tp+=1
                else:
                    fn+=1
                    u.append(i)
            else:
                if(yp[i]>th):
                    fp+=1
                    u.append(i)
                else:
                    tn+=1
                    
                  
        if((tp+fp)!=0):
            pr = tp/(tp+fp)
        else: 
            pr = 0

        if((tp+fn)!=0):
            rc = tp/(tp+fn)
        else: 
            rc = 0

        if(pr+rc!=0):
            f1 = 2*pr*rc/(pr+rc)
        else:
            f1 = 0
        
        d = pd.Series({'TP':tp,'TN':tn,'FP':fp,'FN':fn,'Precision':pr,'Recall':rc,'F1':f1})
            
        return u,d
        
# Load trained Neural Network Model
net = ResNet18(4,False).cuda()
net.load_state_dict(torch.load('../Models/Model4b.pth.tar'))

# Load Random Forest Classifier trained Model
clf = pickle.load(open('../Models/ML1.sav', 'rb'))

# Load Metadata
df = pd.read_csv('../Mets/FinalTest.csv')
df = df[df.columns[1:]]
DL = dl(df,'')
X = genx(net,DL)

# Test Model
print('COVID-19 vs all:')
u, testrep1 = test(df['COVID-19'].values, clf.predict(X))
print(testrep1)
print(u)
                   
print('COVID-19 vs Pneumonia:')
df1 = reindex(pd.concat([ df[df['COVID-19']==1] ,df[df['Pneumonia']==1] ]))
DL1 = dl(df1,'')
u, testrep2 = test(df1['COVID-19'].values, clf.predict(genx(net,DL1)))
print(testrep2)
print(u)

print('COVID-19 vs Others:')
df2 = reindex(pd.concat( [ df[df['COVID-19']==1], df[df['Others']==1] ] ))
DL2 = dl(df2,'')
u, testrep3 = test(df2['COVID-19'].values, clf.predict(genx(net,DL2)))
print(testrep3)
print(u)

dfout = pd.DataFrame()
dfout['Parameters'] = list(testrep3.keys())
dfout['COVID-19 vs Other All'] = list(testrep1.values())
dfout['COVID-19 vs Other Pneumonia'] = list(testrep2.values())
dfout['COVID-19 vs 14 other non-pneumonia diseases'] = list(testrep3.values())
dfout.to_csv('../Losses/ML1test_eval.csv')
print(dfuot)

print('Saved as csv file at: ','../Losses/ML1test_eval.csv')
