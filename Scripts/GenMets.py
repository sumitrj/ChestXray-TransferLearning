# Imports 

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import os

# General Functions

def reindex(df):
  df['Index'] = np.arange(len(df))
  return df.set_index('Index')

# Dataset1:

def gen1():
  df1 = pd.read_csv('../input/rsna-pneumonia-detection-challenge/stage_2_detailed_class_info.csv')
  df1 = df1[df1['class']!='No Lung Opacity / Not Normal']
  df1['Opaque'] = np.array([float('Opacity' in i) for i in df1['class']])
  df1['Normal'] = np.ones(len(df1)) - df1['Opaque'].values
  df1['filename'] = df['patientId']
  df['Pneumonia'] = df['Opaque']
  df1 = df1[['filename', 'Normal', 'Pneumonia']]
  df1 = reindex(shuffle(pd.concat([ df1[df1['Normal']==1],df1[df1['Pneumonia']==1].iloc[:len(df1[df1['Normal']==1])]])))
  df1.to_csv('../Mets/Train1.csv')

# Dataset2:

# Although the author of the dataset containing COVID-19 images claims to have 123 frontal COVID-19 positive CXR images, 
# the github repository shared by the author has more images, owing to very recent updates, modality and repetition
# We have chosen 127 CXR images each for labels of COVID-19, Bacterial Pneumonia, Viral Pneumonia and Normal.

def gen2():
  
  df = pd.read_csv('../../input/covid-chestxray-dataset/metadata.csv')
  df = df[df['finding']=='COVID-19']
  df = df[df['modality'] == 'X-ray']
  cims = df['filename']
  ux = np.random.randint(1,len(ciml),127)
  covid_ims = []
  for i in ux:
    covid_ims.append(ciml[i])
  covid_ims = ['../../input/covid-chestxray-dataset/images/' + i for i in covid_ims]
  
  pims = os.listdir('../../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')
  pims = ['../../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/' + i for i in pims]
  vims = []
  bims = []
  for i in pims:
    if('bacteria' in i):
      bims.append(i)
    if('virus' in i):
      vims.append(i)
  
  viral_ims = []
  ux = np.random.randint(1,len(ciml),127)
  for i in ux:
    viral_ims.append(vims[i])
  
  bacterial_ims = []
  ux = np.random.randint(1,len(ciml),127)
  for i in ux:
    bacterial_ims.append(bims[i])
  
  nims = os.listdir('../../input/chest-xray-pneumonia/chest_xray/train/NORMAL')
  nims = ['../../input/chest-xray-pneumonia/chest_xray/train/NORMAL/' + i for i in nims]
  ux = np.random.randint(1,len(ciml),127)
  normal_ims = []
  for i in ux:
    normal_ims.append(nims[i])
  
  df = pd.DaraFrame()
  df['filename'] = covid_ims + viral_ims + bacterial_ims + normal_ims
  df['COVID-19'] = list(np.ones(127)) + list(np.zeros(127*3))
  df['Viral'] = list(np.zeros(127)) + list(np.ones(127)) + list(np.zeros(127*2))
  df['Bacterial'] = list(np.zeros(127*2)) list(np.ones(127)) + list(np.zeros(127))
  df['COVID-19'] = list(np.zeros(127*3)) + list(np.ones(127)) 
  
  #Split into train and test:  
  df_train = reindex(pd.concat([df[df[i]==1].iloc[:84] for i in df.columns]))
  df_test = reindex(pd.concat([df[df[i]==1].iloc[84:] for i in df.columns]))
  
  df_train.to_csv('../Mets/Train2.csv')
  df_test.to_csv('../Mets/Test2.csv')
  
# Dataset3:

def gen3():
  df2 = pd.read_csv('../../input/data/Data_Entry_2017.csv')
  df2 = df2[['Image Index', 'Finding Labels']]
  df2['filename'] = df1['Image Index']
  
  u = []
  
  for i in df2['filename'][:4999]:
      u.append('../input/data/images_001/images/' + i)

  for i in df1['filename'][4999:10000 + 4999]:
      u.append('../input/data/images_002/images/' + i)

  for i in df1['filename'][10000 + 4999:20000 + 4999]:
      u.append('../input/data/images_003/images/' + i)

  for i in df1['filename'][20000 + 4999:30000 + 4999]:
      u.append('../input/data/images_004/images/' + i)

  for i in df1['filename'][30000 + 4999:40000 + 4999]:
      u.append('../input/data/images_005/images/' + i)

  for i in df1['filename'][40000 + 4999:50000 + 4999]:
     u.append('../input/data/images_006/images/' + i)

  for i in df1['filename'][50000 + 4999:60000 + 4999]:
      u.append('../input/data/images_007/images/' + i)

  for i in df1['filename'][60000 + 4999:70000 + 4999]:
      u.append('../input/data/images_008/images/' + i)

  for i in df1['filename'][70000 + 4999:80000 + 4999]:
      u.append('../input/data/images_009/images/' + i)
  
  u = ['../' + i for i in u]
  df11 = df1.iloc[:80000 + 4999]
  df11['filename'] = u
  df11 = df11[['filename','Finding Labels']] 
  dfn = df11[df11['Finding Labels']=='No Finding']
  df11 = df11[df11['Finding Labels']!='No Finding']
  
  # Shuffle Multiple times
  df11 = reindex(shuffle(shuffle(shuffle(shuffle(df11)))))
  dfcx_up = df11.iloc[:7750]
  
  pn = os.listdir('../../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA')
  pn = ['../../input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/' + i for i in pn]
  dfn = reindex(dfn)
  ns = dfn['filename'][:7750]
  
  nst, otht, pnt = ns[5600:], oth[5600:], pn[2800:]
  ns, oth, pn = ns[:5600], oth[:5600], pn[:2800]
  
  df3_train = pd.DataFrame()
  df3_train['filename'] = list(ns) + list(oth) + list(pn)
  df3_train['Normal'] = list(np.ones(len(ns))) + list(np.zeros(len(oth))) + list(np.zeros(len(pn))) 
  df3_train['Others'] = list(np.zeros(len(ns))) + list(np.ones(len(oth))) + list(np.zeros(len(pn)))
  df3_train['Pneumonia'] = list(np.zeros(len(ns))) + list(np.zeros(len(oth))) + list(np.ones(len(pn)))
  
  df3_test = pd.DataFrame()
  df3_test['filename'] = list(nst) + list(otht) + list(pnt)
  df3_test['Normal'] = list(np.ones(len(nst))) + list(np.zeros(len(otht))) + list(np.zeros(len(pnt))) 
  df3_test['Others'] = list(np.zeros(len(nst))) + list(np.ones(len(otht))) + list(np.zeros(len(pnt)))
  df3_test['Pneumonia'] = list(np.zeros(len(nst))) + list(np.zeros(len(otht))) + list(np.ones(len(pnt)))
  
  df3_train.to_csv('../Mets/Train3.csv')
  df3_test.to_csv('../Mets/Test3.csv')
  
# Dataset4:

def gen4():
  
  c1ims = []
  dc1 = pd.read_csv('../Mets/Train2.csv')
  if(dc1.columns[0]!='filename'):
    dc1 = dc1[dc1.columns[1:]]
  l1 = list(dc1[dc1['COVID-19']==1]['filename'].values)
  dc1 = pd.read_csv('../Mets/Test2.csv')
  if(dc1.columns[0]!='filename'):
    dc1 = dc1[dc1.columns[1:]]
  l1 += list(dc1[dc1['COVID-19']==1]['filename'].values)   
  
  l2 = os.listdir('../../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19')
  l2 = ['../input/covid19-radiography-database/COVID-19 Radiography Database/COVID-19/' + i for i in l2]
  
  c1ims = l1 + l2
  cim_train = cims[:-25]
  cim_test = cims[-25:]
  
  # Note: Even if we are considering images from 'Test2', there would be no problem in validation. 
  #       The images used to validate predictions finally are in cim_test.
  #       The images used in cim_test are the last 25 images of the dataset 'COVID-19 Radiography Database' which is being used for the first time in our datasets.
  
  # Take 1000 images each for labels Pneumonia, Normal and Others from Train3 set for Train4
  df_non-cov_train = pd.read_csv('../Mets/Train3.csv')
  if(df_non-cov_train.columns[0]!='filename'):
    df_non-cov_train = df_non-cov_train[df_non-cov_train.columns[1:]]
  df_non-cov_train = df_non-cov_train[df_non-cov_train.columns[1:]]
  df_non-cov_train = reindex(pd.concat([df_non-cov_train[df_non-cov_train[i]==1].iloc[:1000] for i in df_non-cov_train.columns[1:]]))
  
  df_ML_train = pd.DataFrame()
  df_non-cov_ML_train = reindex(pd.concat([df0[df0[i]==1].iloc[:242] for i in df_non-cov_train.columns[1:]]))
  df_ML_train['filename'] = list(df_non-cov_ML_train['filename'].values) + cim_test[:242]
  df_ML_train['Normal'] = list(np.ones(242)) + list(np.zeros(242*2+242)) 
  df_ML_train['Others'] = list(np.zeros(242)) + list(np.ones(242)) + list(np.zeros(242+242))
  df_ML_train['Pneumonia'] = list(np.zeros(242*2)) + list(np.ones(242)) + list(np.zeros(242))
  df_ML_train['COVID-19'] = list(np.zeros(242*3)) + list(np.ones(242))
  
  df_ML_train.to_csv('Train4a.csv')
  
  df4_train = pd.DataFrame()
  df4_train['filename'] = list(df_non-cov_train['filename'].values) + cim_train
  df4_train['Normal'] = list(np.ones(1000)) + list(np.zeros(1000*2+321)) 
  df4_train['Others'] = list(np.zeros(1000)) + list(np.ones(1000)) + list(np.zeros(1000+321))
  df4_train['Pneumonia'] = list(np.zeros(1000*2)) + list(np.ones(1000)) + list(np.zeros(321))
  df4_train['COVID-19'] = list(np.zeros(3000)) + list(np.ones(321))
  df4_train.to_csv('Train4b.csv')
  
  # Take 50 images each for labels Pneumonia, Normal and Others from Train3 set for Test4
  df_non-cov_test = reindex(pd.concat([df0[df0[i]==1].iloc[1000:1000+50] for i in df0.columns[1:]])) 
  df4_test = pd.DataFrame()
  df4_test['filename'] = list(df_non-cov_test['filename'].values) + cim_test
  df4_test['Normal'] = list(np.ones(50)) + list(np.zeros(50*2+25)) 
  df4_test['Others'] = list(np.zeros(50)) + list(np.ones(50)) + list(np.zeros(50+25))
  df4_test['Pneumonia'] = list(np.zeros(50*2)) + list(np.ones(50)) + list(np.zeros(25))
  df4_test['COVID-19'] = list(np.zeros(150)) + list(np.ones(25))
  df4_test.to_csv('FinalTest.csv')
  
  
gen1()
print('Generated Dataset1')
gen2()
print('Generated Dataset2')
gen3()
print('Generated Dataset3')
gen4()
print('Generated Dataset4 and Test Dataset')
