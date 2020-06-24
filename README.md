# WACV Paper 220 Supplementary Material

## Setting up Environment

### Docker container and python libraries set-up

To avoid the hassle of setting up environment by installing individual libraries, we recommend the use of [NGC PyTorch container](https://ngc.nvidia.com/registry/nvidia-pytorch) since it has all the libraries preinstalled in it. Additionally, the python library [pydicom]([https://pypi.org/project/pydicom/](https://pypi.org/project/pydicom/))  would be needed to work with .dcm images. Install it by running the following in the container.
	
	pip3 install pydicom

## Downloading Datasets
The datasets used by us are from Kaggle. To download the datasets in a headless setting, Kaggle API's need to be used. [Kaggle API Documentation](https://github.com/Kaggle/kaggle-api) describes the process of obtaining credentials. The credeentials can be downloaded in a json file which needs to be added to the root directory. An easy way to do it is to download it on your system, add it to a github repository, clone the repository in the container and move it to the rooth directory.

**The following datasets would be needed:**
1) [Paul and Cohen's COVID-19 positive CXR image collection ](https://github.com/ieee8023/covid-chestxray-dataset)
2) [COVID-19 radiography database](https://www.kaggle.com/tawsifurrahman/covid19-radiography-database)
3) [RSNA's Pneumonia detection challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge)
4) [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) 
5) [NIH's ChestXray14 dataset](https://www.kaggle.com/nih-chest-xrays/data)

**Download commands:**
After setting up credentials,
Make a directory 'input' in workspace:

	mkdir input

Inside the input directory, make directories for the indivdual datasets, download the datasets and unzip them.
	
	cd input
	mkdir covid19-radiography-database rsna-pneumonia-detection-challenge chest-xray-pneumonia data 
	git clone https://github.com/ieee8023/covid-chestxray-dataset.git
	
	cd covid19-radiography-database
	kaggle datasets download -d tawsifurrahman/covid19-radiography-database
	unzip covid19-radiography-database.zip
	
	cd ../rsna-pneumonia-detection-challenge
	kaggle datasets download -d rsna-pneumonia-detection-challenge
	unzip rsna-pneumonia-detection-challenge.zip
	
	cd ../chest-xray-pneumonia
	kaggle datasets download -d paultimothymooney/chest-xray-pneumonia
	unzip chest-xray-pneumonia.zip
	
	cd ../data
	kaggle datasets download -d nih-chest-xrays/data
	unzip data.zip
	
	cd ..	

**Clone this repository outside the 'input' directory in the 'workspace' directory**

**Contents of the repository:**

**a) Mets**
Contains csv files of metadata. Each file has a dataframe with the columns 'Index', 'filename', 'label 1', 'label 2', 'label 3'...
For a given row, 
**filename** specifies the filename of a particular image. It could be just the title of the image or in some cases, the location of the image w.r.t the **Scripts** directory.
**'label 1', 'label 2', 'label 3'**  give the onehot encoded label vector for the image.

**b) Scripts**
Has python scripts to geerate metadata files, train and validate Deep Learning and Machine Learning models.

**c) Models**
Has the Deep Learning (PyTorch Models .pth.tar) and Machine Learning models (.sav)

**d) Losses**
Has csv files of training loss decay trends for deep learning models and classification report


## Compiling Metadata

The Metadata has been compiled into csv files and placed in the 'Mets' directory. It has been obtained from various sources mentioned above. Since the problem is very recent, the owners of the above mentioned repositories have been channging the content from time to time. We have begun solving the problem from around 3 months, hence there might be slight changes when you run the program which generates the Metatdata files. The following lets you generates the files as per requirement.

Navigate to the Scripts directory by
	
	cd Scripts

Running the file GenMets.py would generate Train and Test Metadata required for building the feature extractor and classifier
	
	python3 GenMets.py

### Description of Datasets generated:

**a) Dataset1:**
Data of Pneumonia positive Chest X-ray (CXR) images and Normal CXR images.

	Train: 8851 CXR images each for classes Normal and Opaque.  
	Test: All images were used for training. Since Binary classification of pneumonia wasn't the ulitmate goal, the model needn't be evaluated and hence there was no need of test data.

**b) Dataset2:**
Data of COVID-19 positive CXR images,  Viral Pneumonia positive CXR images, Bacterial Pneumonia positive CXR images and Normal CXR images.
	
	Train: 84 CXR images each for classes COVID-19, Viral pneumonia,Bacterial pneumonia and Normal
	Test: 44 CXR images each for classes COVID-19, Viral pneumonia,Bacterial pneumonia and Normal

**c) Dataset3:**
Data of Pneumonia positive CXR images, CXR images positive of 14 diseases and Normal CXR images.

	Train: 2800 CXR images of Pneumonia, 5600 CXR images of Normal, 5600 CXR images of Other diseases from CXR14 
	Test: 1075 CXR images of Pneumonia, 2150 CXR images of Normal, 2150 CXR images of Other diseases from CXR14

**d) Dataset4a:**
Data of COVID-19 positive CXR images, Pneumonia positive CXR images, and CXR images positive of 14 diseases and Normal CXR images.

	Train: 242 CXR images each of Pneumonia, Normal, Other diseases, COVID-19 labels
	
**e) Dataset4b:**
Data of COVID-19 positive CXR images, Pneumonia positive CXR images, and CXR images positive of 14 diseases and Normal CXR images.

	Train: 1000 CXR images each of Pneumonia, Normal and Other diseases, 321 COVID-19 positive CXR images 
	Test (FinalTest): 50 CXR images each of Pneumonia, Normal and Other diseases, 25 COVID-19 positive CXR images

## Training ResNet18 Model to build a feature extractor:

The scripts Train1.py, Train2.py, Train3.py and Train4.py are to be executed to build the feature extractor.

A general description of the prompts is given below:

On running any of the above mentioned scripts, the following will prompt:

	Load Trained? 0

In case of Part 1, where we are using a ResNet18 model pretrained on ImageNet dataset, entering 0 here will load a ResNet18 model pretrained on ImageNet dataset as basemodel.

In case of Parts 2,3, and 4, the Models Model1, Model2 and Model3 are used as basemodels. These models are desribed in the subsequent subsections.

Entering 1 here will ask you for the path of the model which is to be used as basemodel.

	Load Trained? 1
	Enter Base Model Path: ../Models/<BaseModelname>.pth.tar

**Enter 0 here and press enter for all of the parts**

You'll be then prompted to enter Model Name and Number of epochs.

	Enter Model Name: Modelx <example>
	Enter Number of epochs: 700 <example>

After execution, a file named 'ModelxLosses.csv'(example) will be stored in the 'Losses' directory and a file named 'Modelx.pth.tar' will be stored in 'Models' directory.


### Part 1: Training over Dataset1

We have used ResNet18 model pretrained on ImageNet dataset. We train this model to build a binary classifier of CXR images into labels 'Pneumonia' and 'Normal'. The model trains for 700 epochs with Sigmoid Activation function,  Stochastic Gradient Descent Optimizer and Categorical Cross Entropy Loss Function. 

Run the script 'Train1.py' from the 'Scripts' directory to obtain the model. Enter 0 for Load Trained, Model1 for model name and 700 for number of epochs. This generates Model1.pth.tar in 'Models' directory and and 'Model1Losses.csv' in Losses Directory.

### Part 2: Training Over Dataset 2

We use Model1 as base model to build a classifier of CXR images into labels 'COVID-19', 'Other Pneumonia' and 'Normal'. The model trains for 190 epochs with ReLU Activation function,  Adam Optimizer and WCAC Loss Function. 

Run the script 'Train2.py' from the 'Scripts' directory to obtain the model. Enter 0 for Load Trained, Model2 for model name and 190 for number of epochs. This generates Model2.pth.tar in 'Models' directory and and 'Model2Losses.csv' in Losses Directory.

### Part 3: Training Over Dataset 3

We use Model2 as base model to build a classifier of CXR images into labels 'Other diseases', 'Pneumonia' and 'Normal'. The model trains for 400 epochs with ReLU Activation function, Adam Optimizer and WCAC Loss Function. 

Run the script 'Train3.py' from the 'Scripts' directory to obtain the model. Enter 0 for Load Trained, Model3 for model name and 400 for number of epochs. This generates Model3.pth.tar in 'Models' directory and and 'Model3Losses.csv' in Losses Directory.

### Part 4a: Training Over Dataset 4a

We use Model3 as base model to build a classifier of CXR images into labels 'COVID-19', 'Pneumonia', 'Other diseases', and 'Normal'. The model trains for 290 epochs with ReLU Activation function, Adam Optimizer and WCAC Loss Function. 

Run the script 'Train4a.py' from the 'Scripts' directory to obtain the model. Enter 0 for Load Trained, Model4a for model name and 290 for number of epochs. This generates Model4a.pth.tar in 'Models' directory and and 'Model4aLosses.csv' in Losses Directory.

### Part 4b: Training Over Dataset 4b

We use Model4a as base model to build a classifier of CXR images into labels 'Other diseases', 'Pneumonia' and 'Normal'. The model trains for 250 epochs with ReLU Activation function, Adam Optimizer and WGCAC Loss Function.

Run the script 'Train4b.py' from the 'Scripts' directory to obtain the model. Enter 0 for Load Trained, Model4b for model name and 250 for number of epochs. This generates Model4b.pth.tar in 'Models' directory and and 'Model4bLosses.csv' in Losses Directory.

**Intermediate models Model1.pth.ta, Model4a.pth.tar and Model3.pth.tar have been removed to bring total repository size under 100MB** 
**Final model, i.e Model4b.pth.tar is still present**

## Training the Classifier
Random Forest classifier is trained over the features extracted by Model4b.
Run the following in the 'Scripts' directory:
	
	python3 ML1train.py
On completion a file 'ML1.sav' will be generated in the 'Models' directory which stores the Random Forest Classifier Model

## Evaluating Model performance on Test Data
Run the following in the 'Scripts' directory:
	
	python3 ML1test.py
On completion a file 'ML1test_eval.csv' will be generated in the 'Losses' directory which has the evaluation parameters for the classifier.

## Results:

	COVID-19 vs Pneumonia:
	TP           23.000000
	TN           47.000000
	FP            1.000000
	FN            2.000000
	Precision     0.958333
	Recall        0.920000
	F1            0.938776

	COVID-19 vs non-Pneumonia-CX14 diseases:
	TP           23.000000
	TN           47.000000
	FP            4.000000
	FN            2.000000
	Precision     0.851852
	Recall        0.920000
	F1            0.884615
