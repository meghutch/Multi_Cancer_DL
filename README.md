# Multi_Cancer_DL
Deep learning cancer classification using methylation profiles

This repository contains the code to construct deep learning neural networks that can take methylation data as inputs and predict cancer type. 

01_Data_Processing: Contains the initital scripts to clean data

03_Analysis: Scripts to carry out different analyses on the Full Cohort (7 different disease statuses) or the GI cohort (5 disease statuses). 

scripts: .py and sh batch scripts used to run code on server

03_Analysis Details:

PCA scripts:

* We performed Principal Component Analysis and identified the top PCs that explained 90% of the variance (only performed on Full Cohort)

Logistic Regression Scripts:

* Scripts to run logistic regression analysis to provide benchmark classification performance to compare Neural Network performance

Random Forest scripts:

* Featurization scripts indicate scripts run on the training set where features from the Variable Importance table were abstracted and included in neural network analyses

* Classifier/Classification scripts indicate scripts trained on the training set and then testing set to provide benchmark classification performance to compare Neural Network performance

Neural Network scripts:

* Clinical in the script name indicates that (gender, age, mean fragment length) were inlcuded

* RF in the script name indicates that features from random forest (abstracted from featurization ipynb scripts) were included 
