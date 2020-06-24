# Imports

import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import cv2
from scipy.ndimage import gaussian_filter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torchvision.models as models

from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier


# General Functions

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()
        
    def forward(self, x):
        return x

def genx(Model,DL):
    
    Model.net.fc = Identity()
    X = []
    for i in range(DL.__len__()):
        X.append(Model(DL.__getitem__(i)[0]).detach().cpu().numpy())
        #print(i)
    X = np.array(X)
    X2 = X.reshape(X.shape[0],X.shape[-1])
    return X2
    
def ML(X,y):
    
    Xtr, Xt, ytr, yt = train_test_split(X,y, test_size = 0.3, train_size = 0.7)
    clf = RandomForestClassifier(n_estimators=4000, max_depth=40,random_state=0)
    clf.fit(Xtr,ytr)
    yp = clf.predict(Xt)
    return clf, test(yt,yp,0.5)    
    
def test(yt,yp,th):
       
        tp,tn,fp,fn = 0,0,0,0
        
        for i in range(len(yt)): 
            if(yt[i]):
                if(yp[i]>th):
                    tp+=1
                else:
                    fn+=1
            else:
                if(yp[i]>th):
                    fp+=1
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
      
        return d

def htg(im):
    im = cv2.equalizeHist(im)
#    img = gaussian_filter(im,sigma=25)
#    im = im*(img>50)
    return im

def imP3(im):
    
    im1,im2,im3 = im[:,:,0],im[:,:,1],im[:,:,2]
    im[:,:,0],im[:,:,1],im[:,:,2] = htg(im1), htg(im2), htg(im3)
    im = cv2.resize(im,(103,128))
    return im

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

def DLout(model,DL):
    yp = []
    yt = []

    for i in range(DL.__len__()):
        image, label = DL.__getitem__(i)
        out = model(image)
        yp.append(out)
        yt.append(label)
    
    return (yt,yp)
    
def dl(df):
    return DataLoader(df,transforms.Compose([transforms.Normalize, transforms.RandomCrop, transforms.RandomSizedCrop]))

def comp(x):
    return (np.ones(len(x)) - x)

def reindex(df):
    df['Index'] = np.arange(len(df))
    return df.set_index('Index')
    
# Read Metadata

df = pd.read_csv('../Mets/Test4.csv')
df = df[df.columns[1:]]
DL = dl(df)

# Load Model

class ResNet18(nn.Module):

    def __init__(self, classCount, isTrained):
        
        super(ResNet18, self).__init__()
        self.net = models.resnet18(pretrained=isTrained)
        kernelCount = self.net.fc.in_features
        self.net.fc = nn.Sequential(nn.Linear(kernelCount, classCount))

    def forward(self, x):
        x = self.net(x)
        return F.relu(x)
        
nc = int(input('Enter number of output units of Network: '))        
n3 = ResNet18(nc,False).cuda()
n3.load_state_dict(torch.load(input('Enter model path ')))

print('Path available ', n3(DL.__getitem__(0)[0])[0].shape, DL.__getitem__(0)[1].shape) 

# Predict Output
if(int(input('Check NN? '))):
    
    diss = df[df.columns[1:]]
    print('Neural Network Classifier: ')
    ytcx,ypcx = DLout(n3,DL)
    lt, lp = np.array([i.cpu().numpy() for i in ytcx]) , np.array([i[0].detach().cpu().numpy() for i in ypcx])
    yts, yps = [lt[:,i] for i in range(nc)], [lp[:,i] for i in range(nc)]

    d01 = dict(zip(np.arange(nc), [[] for i in range(nc)]))

    for i in range(nc):
        for k in range(150):
            j=k/100
            d01[i].append(test(yts[i],yps[i],j)['F1'])

    t01 = dict(zip(np.arange(nc), [[] for i in range(nc)]))
    for i in d01:
        print(diss[i], np.argmax(d01[i])/100, np.max(d01[i]))
        t01[i] = np.argmax(d01[i])/100

if(int(input('Train ML Classifier? '))):
    
    na = input('Enter ML Model Title: ' )
    df1 = pd.read_csv('../Mets/Met4.csv')
    DL1 = dl(df1)
    X = genx(n3,DL1)
    clf, rep = ML(X,df1['COVID-19'].values)
    
    
    Xt = genx(n3,DL)
    rep = test(df['COVID-19'].values,clf.predict(Xt),0.5)
    rep.to_csv('../Losses/COVIDTest4rep.csv')                        
    print(rep)
    pickle.dump(clf, open('../Models/T4/' + na, 'wb'))
