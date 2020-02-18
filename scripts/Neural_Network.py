#!/usr/bin/env python
# coding: utf-8

# # **Methylation Biomarkers for Predicting Cancer**
# 
# ## **Deep Learning Approaches for Cancer Classification**
# 
# **Author:** Meg Hutch
# 
# **Date:** February 14, 2020
# 
# **Objective:** Use neural networks to classify cancer type. 
# 
# **Note:** In this version, I will only test the ability of methylation levels to classify cancer types. I will not include phenotypic data for now. Additionally, this version has our data split 70% for training and 30% for testing. The 70% training data will undergo leave-one-out-cross-fold validation to tune hyperparameters prior to testing final performance on the 30% test set. 
# 
# Note: This is the new version of the script where we normalize gene counts using DEseq2 in the initial pre-processing script in R. This provided more than double the number of Principal Components that make up 90% of the variance (157). Regardless, we will begin running the deep learning classifier on the revised data. 

# In[1]:


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors
import seaborn as sns


# In[2]:


# set working directory for git hub
import os
os.chdir('/projects/p31049/Multi_Cancer_DL/')
#os.chdir('C:\\Users\\User\\Box Sync/Projects/Multi_Cancer_DL/')
os. getcwd()


# **Import Training, Testing, and Principal component data**

# In[3]:


# Training set
mcTrain = pd.read_csv('02_Processed_Data/mcTrain_70_30.csv')
# Testing set
mcTest = pd.read_csv('02_Processed_Data/mcTest_70_30.csv')
# Principal Components that make up 90% of the variance of the training set
genesTrain_transformed_90 = pd.read_csv('02_Processed_Data/genesTrain_transformed_157pc_70_30.csv')
# Principal Components projected onto the test set
genesTest_transformed_90 = pd.read_csv('02_Processed_Data/genesTest_transformed_157pc_70_30.csv')


# In[4]:


mcTrain.head()
genesTrain_transformed_90.head()


# # **Pre-Process Data**

# In[5]:


# remove genetic data from the mcTrain dataset
mcTrain = mcTrain[['seq_num','diagnosis', 'dilute_library_concentration', 'age', 'gender', 'frag_mean']]

# do the same for the testing set
mcTest = mcTest[['seq_num','diagnosis', 'dilute_library_concentration', 'age', 'gender', 'frag_mean']]


# In[6]:


# rename the first column name of the PC dataframes
genesTrain_transformed_90.rename(columns={'Unnamed: 0':'seq_num'}, inplace=True)
genesTest_transformed_90.rename(columns={'Unnamed: 0':'seq_num'}, inplace=True)


# In[7]:


# merge PCs with clinical/phenotypic data
mcTrain = pd.merge(mcTrain, genesTrain_transformed_90, how="left", on="seq_num") 
mcTest = pd.merge(mcTest, genesTest_transformed_90, how="left", on="seq_num") 


# **Shuffle the training and test sets**
# 
# Currently, all disease states are in order - we don't want to feed to the network in order!

# In[8]:


import random
random.seed(222020)
mcTrain = mcTrain.sample(frac=1, axis = 0).reset_index(drop=True) # frac = 1 returns all rows in random order
mcTest = mcTest.sample(frac=1, axis = 0).reset_index(drop=True)


# **Create a new numeric index and drop seq_num and demographic data for these experiments**
# 
# For future code we want the index to be numeric

# In[9]:


# Create new ids
mcTrain['id'] = mcTrain.index + 1
mcTest['id'] = mcTest.index + 243

# Drop num_seq
mcTrain = mcTrain.drop(columns=["seq_num", "dilute_library_concentration", "age", "gender", "frag_mean"])
mcTest = mcTest.drop(columns=["seq_num", "dilute_library_concentration", "age", "gender", "frag_mean"])


# **Remove Labels (Diagnosis) from the datasets**

# In[10]:


mcTrain_x = mcTrain.drop(columns=["diagnosis"])
mcTest_x = mcTest.drop(columns=["diagnosis"])


# **Create Labeled Datasets**

# In[11]:


mcTrain_y = mcTrain[['id','diagnosis']]
mcTest_y = mcTest[['id','diagnosis']]


# In[12]:


# Examine the unique target variables
mcTrain_y.diagnosis.unique()


# In[13]:


# Replace each outcome target with numerical value
mcTrain_y = mcTrain_y.replace('HEA', 0)
mcTrain_y = mcTrain_y.replace('CRC', 1)
mcTrain_y = mcTrain_y.replace('ESCA', 2)
mcTrain_y = mcTrain_y.replace('HCC', 3)
mcTrain_y = mcTrain_y.replace('STAD', 4)
mcTrain_y = mcTrain_y.replace('GBM', 5)
mcTrain_y = mcTrain_y.replace('BRCA', 6)

mcTest_y = mcTest_y.replace('HEA', 0)
mcTest_y = mcTest_y.replace('CRC', 1)
mcTest_y = mcTest_y.replace('ESCA', 2)
mcTest_y = mcTest_y.replace('HCC', 3)
mcTest_y = mcTest_y.replace('STAD', 4)
mcTest_y = mcTest_y.replace('GBM', 5)
mcTest_y = mcTest_y.replace('BRCA', 6)


# **Convert seq_num id to index**

# In[14]:


mcTrain_x = mcTrain_x.set_index('id')
mcTrain_y = mcTrain_y.set_index('id')

mcTest_x = mcTest_x.set_index('id')
mcTest_y = mcTest_y.set_index('id')


# **Normalize Data**
# 
# From my reading, it seems that normalization, as opposed to standardization, is the more optimal approach when data is not normally distributed. 
# 
# Normalization will rescale our values into range of [0,1]. We need to normalize both the training and test sets

# In[15]:


from sklearn.preprocessing import MinMaxScaler

# The normalization function to be performed will convert dataframe into array, for this reason we'll have to convert it back
# Thus, need to store columns and index
# select all columns
cols = list(mcTrain_x.columns.values)
index_train = list(mcTrain_x.index)
index_test = list(mcTest_x.index)

# Normalize data
scaler = MinMaxScaler()
mcTrain_x = scaler.fit_transform(mcTrain_x.astype(np.float))
mcTest_x = scaler.fit_transform(mcTest_x.astype(np.float))

# Convert back to dataframe
mcTrain_x = pd.DataFrame(mcTrain_x, columns = cols, index = index_train)
mcTest_x = pd.DataFrame(mcTest_x, columns = cols, index = index_test)


# # **Downsampling**

# In[16]:


# Subset healthy patients
class0 = mcTrain_y[mcTrain_y.diagnosis == 0]

# Select only 30 healthy patients - we will remove these patients
class0 = class0.head(30)
class0 = class0.index.tolist()

mcTrain_y = mcTrain_y[~mcTrain_y.index.isin(class0)]
print(mcTrain_y.head(20))

# Print the cases we wanted to remove
print(class0)

#Observe class distributions
class0_new = mcTrain_y[mcTrain_y.diagnosis == 0]
print('# Healthy Patients', class0_new.shape) # shoud be 31
print('# Full Training Set', mcTrain_y.shape)


# **Remove excess healthy patients from the input training set and original dataset**

# In[17]:


mcTrain_x = mcTrain_x[~mcTrain_x.index.isin(class0)]
print('# Full Training Set', mcTrain_x)


# # Construct & Run Neural Network

# In[18]:


# Import PyTorch packages
import torch
from torch import nn
#from torchvision import datasets, transforms
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import TensorDataset
import torch.nn.functional as F
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# In[19]:


# define list for results
results_ls = []

# Where we will store correct/incorrect classifications
incorrect_ls = []
correct_ls = []

# Leave-one-out-cross-fold validation function - the for loop will iterate through the dataset, removing one sample (patient)
# at a time in order to create k training and test datasets (where k = number of total samples) always with one sample missing
#for index in range (0, 242):
for index in range (0, 212): # 212 observations when we downsample healthy patients 
    # X - features; 
    mcTrain_xy_drop = mcTrain_x.drop(mcTrain_x.index[index]) # add 'drop'suffix so we can differentiate the df with index and the array that will be created in next line
    mcTrain_xy = np.array(mcTrain_xy_drop, dtype = "float32")
    
    # y - target/outputs
    mcTrain_yz_drop = mcTrain_y.drop(mcTrain_y.index[index]) 
    mcTrain_yz = np.array(mcTrain_yz_drop, dtype = "float32")
    
    # reformat into tensors
    xb = torch.from_numpy(mcTrain_xy)
    yb = torch.from_numpy(mcTrain_yz)
    
    # squeeze - function is used when we want to remove single-dimensional entries from the shape of an array.
    yb = yb.squeeze(1) 
    
    # subset the equivalent test set
    mcTrain_test_x_drop = mcTrain_x.iloc[[index]] # add 'drop'suffix so we can differentiate the df with index and the array that will be created in next line
    mcTrain_test_x = np.array(mcTrain_test_x_drop, dtype = "float32")
            
    # y - targets/outputs
    mcTrain_test_y_drop = mcTrain_y.iloc[[index]]
    mcTrain_test_y = np.array(mcTrain_test_y_drop, dtype = "float32")
        
    # Convert arrays into tensors
    test_xb = torch.from_numpy(mcTrain_test_x)
    test_yb = torch.from_numpy(mcTrain_test_y)
    
    # Define the batchsize
    batch_size = 32

    # Combine the arrays
    trainloader = TensorDataset(xb, yb)
    
    # Training Loader
    trainloader = DataLoader(trainloader, batch_size, shuffle=True)
    
    ## Build the Model and define hyperparameters
    
    # Set parameters for grid search
    #lrs = [1e-2, 1e-3, 1e-4]
    #epochs = [100, 150, 200, 250]
    
    # Define the model with hidden layers
    model = nn.Sequential(nn.Linear(157, 50),
                          nn.ReLU(),
                          nn.Linear(50, 7))
                      
    # Set Stoachastic Gradient Descent Optimizer and the learning rate
    #optimizer = optim.SGD(model.parameters(), lr=0.003)

    # Set Adam optimizer: similar to stochastic gradient descent, but uses momentum which can speed up the actual fitting process, and it also adjusts the learning rate for each of the individual parameters in the model
    optimizer = optim.Adam(model.parameters(), lr=0.01,  weight_decay=0.01) # we can also change momentum parameter

    # loss function
    criterion = nn.CrossEntropyLoss() #don't use with softmax or sigmoid- PyTorch manual indicates "This criterion combines nn.LogSoftmax() and nn.NLLLoss() in one single class."
    
    # Set epochs - number of times the entire dataset will pass through the network
    epochs = 500
    for e in range(epochs):
        # Define running loss as 0
        running_loss = 0
        
        # Run the model for each xb, yb in the trainloader. For the number of epochs specified, the 
        for xb, yb in trainloader:
            # clear gradients - otherwise they are stored
            optimizer.zero_grad()
            # Training pass
            output = model.forward(xb)
            # caluclate loss calculated from the model output compared to the labels
            loss = criterion(output, yb.long()) 
            # backpropagate the loss
            loss.backward()
            # step function to update the weights
            optimizer.step()
        
            running_loss += loss.item() # loss.item() gets the scalar value held in the loss. 
            # += function: Adds the running_loss (0) with loss.item and assigns back to running_loss
        else:
            print("Epoch {}/{}, Training loss: {:.5f}".format(e+1, epochs, running_loss/len(trainloader)))

    # Apply the model to the testing dataset
    # Thus will enable us to see the predictions for each class
    ps = model(test_xb)
    #print('Network Probabilities', ps)
    
    # Obtain the top prediction
    top_p, top_class = ps.topk(1, dim=1)
    #print('top prediction', top_p)
    #print('true vals', test_yb[:10])
        
    # Drop the grad by using detach
    top_p = top_p.detach().numpy()
    top_class = top_class.detach().numpy()

    # convert to integers
    top_class = top_class.astype(np.int)
    test_yb = test_yb.numpy()
    test_yb = test_yb.astype(np.int)
    
    #print('top class', top_class[:10])
    #print('prediction:', top_class)
    #print('true:', test_yb)
                
    # compare top_class to test_yb
    if top_class == test_yb:                
        results = 1 # prediction and true value are equal
    else: 
        results = 0
    
    # Create if-else statements to identify which classes are being classified correctly/incorrectly
    if results == 0:
        incorrect = test_yb
    else: 
        incorrect = np.array([[999]], dtype=int)
        
    if results == 1:
        correct = test_yb
    else: 
        correct = np.array([[999]], dtype=int)
    #print('Results:', results)
    
    results_ls.append(results)
    incorrect_ls.append(incorrect)
    correct_ls.append(correct)
    print(results_ls) 


# # **Print out description of Experiment**

# In[21]:


print('Hidden Layers: 50, Weight Decay: 0.10, LR: 0.01 , Epochs: 500')


# # **Determine LOOCV Mean Error**

# In[22]:


percent_correct = sum(results_ls)
percent_correct = percent_correct/len(mcTrain_y)*100
percent_incorrect = 100 - percent_correct
print('Percent Error', round(percent_incorrect, 1))


# # **Incorrect Predictions**

# In[35]:


## Remove the correct elements from the ls to faciliate transforming this list into a dataframe
# First, concatenate all incorrect list elements and format into dataframe
incorrect_res = np.concatenate(incorrect_ls)
incorrect_res = pd.DataFrame(incorrect_res)
incorrect_res.columns = ['diagnosis']
incorrect_res = incorrect_res[incorrect_res.diagnosis != 999] # 999 are the results that were correct - we remove these

# Count number of incorrect predictions by diagnosis
incorrect_pred = incorrect_res.groupby(['diagnosis']).size()
incorrect_pred = pd.DataFrame(incorrect_pred)
incorrect_pred.columns = ['Count']

# Convert the index to the first column and change the numebr to categorical variables
incorrect_pred.reset_index(level=0, inplace=True)
incorrect_pred['diagnosis'] = incorrect_pred['diagnosis'].map({0: 'HEA', 1: 'CRC', 2: 'ESCA', 3: 'HCC', 4: 'STAD', 5:'GBM', 6:'BRCA'})

# Add a column with the number of cases in each class
mcTrain_y['diagnosis'] = mcTrain_y['diagnosis'].map({0: 'HEA', 1: 'CRC', 2: 'ESCA', 3: 'HCC', 4: 'STAD', 5:'GBM', 6:'BRCA'})
class_size = mcTrain_y.groupby(['diagnosis']).size()
class_size = pd.DataFrame(class_size)
class_size.columns = ['Sample_n']

# bind class_size to the pred df diagnoses
incorrect_pred = pd.merge(incorrect_pred, class_size, how="left", on="diagnosis") 

# Calculate the percent error for each class
incorrect_pred['Count_Perc_Incorrect'] = incorrect_pred['Count']/incorrect_pred['Sample_n']
incorrect_pred['Count_Perc_Incorrect'] = incorrect_pred['Count_Perc_Incorrect'].multiply(100)

print(round(incorrect_pred, 1))


# # **Correct Predictions**

# In[36]:


## Remove the incorrect elements from the ls to faciliate transforming this list into a dataframe
# First, concatenate all incorrect list elements and format into dataframe
correct_res = np.concatenate(correct_ls)
correct_res = pd.DataFrame(correct_res)
correct_res.columns = ['diagnosis']
correct_res = correct_res[correct_res.diagnosis != 999] # 999 are the results that were incorrect - we remove these

# Count number of correct predictions by diagnosis
correct_pred = correct_res.groupby(['diagnosis']).size()
correct_pred = pd.DataFrame(correct_pred)
correct_pred.columns = ['Count']

# Convert the index to the first column and change the numebr to categorical variables
correct_pred.reset_index(level=0, inplace=True)
correct_pred['diagnosis'] = correct_pred['diagnosis'].map({0: 'HEA', 1: 'CRC', 2: 'ESCA', 3: 'HCC', 4: 'STAD', 5:'GBM', 6:'BRCA'})

# Add a column with the number of cases in each class
class_size = mcTrain_y.groupby(['diagnosis']).size()
class_size = pd.DataFrame(class_size)
class_size.columns = ['Sample_n']

# bind class_size to the pred df diagnoses
correct_pred = pd.merge(correct_pred, class_size, how="left", on="diagnosis") 

# Calculate the percent correct for each class
correct_pred['Count_Perc_Correct'] = correct_pred['Count']/correct_pred['Sample_n']
correct_pred['Count_Perc_Correct'] = correct_pred['Count_Perc_Correct'].multiply(100)

print(round(correct_pred, 1))


# # **Save Predictions**

# In[ ]:


#correct_pred.to_csv('/home/mrh1996/Multi_Cancer_DL/04_Results/correct_preds.csv')
#incorrect_pred.to_csv('/home/mrh1996/Multi_Cancer_DL/04_Results/incorrect_preds.csv')

