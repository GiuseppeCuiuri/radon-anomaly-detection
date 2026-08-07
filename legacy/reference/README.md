# RADON: Robust Autoencoder for Unsupervised Anomaly Detection
This repository contains the code of the contribution RADON. The  structure of the repository is as follows:
 - ./Train_models : save in this repository your models after training 
 - ./RADON.py : RADON algorithm is defined in this file. If you want to modify the structure of the autoencoder, you can modify the class RADON part.
 - ./main.ipynb : the training and testing jupyter notebook file 
 - ./radon.yml : this file contains all the packages required to run main.ipynb
 - ./readme.txt  
 
 # Installation 
 To install the required depndencies with conda, run the following commands: 
 
 conda env create -n radon -f radon.yml 
 conda activate radon  
 
 
 # Remark: 
 
- The training and testing data must be in ../data.
- In main.ipynb, we defined the default hyperparameters of our algorithm.
- Modify the data processing steps if you have another data processing pipeline. 
 
