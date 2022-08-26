# Mod13_Challenge
# UW-finctech-2022
This is  a public repo for the Module 13 Challenge of the UW Fintech Bootcamp in 2022.


## Technologies and Libraries

Jupyter lab
pandas. 1.3.5
scikit-learn 1.0.2
Tensorflow

## Installation Guide

Install jupyter lab by running the command jupyter lab in your terminal

Install the following dependencies an dmocdules from the libraries above

```
  import pandas as pd
  from pathlib import Path
  import tensorflow as tf
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.models import Sequential
  from sklearn.model_selection import train_test_split
  from sklearn.preprocessing import StandardScaler,OneHotEncoder

```

## Overview of the analysis

* Purpose of the analysis

 Alphabet Soup’s business team receives many funding applications from startups every day. This team has asked you, a risk management associate at the company(Assumed role), to help them create a model that predicts whether applicants will be successful if funded by Alphabet Soup.

* Financial Information on the data * 

The datase exists as a CSV file containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. The CSV file contains a variety of information about each business, including whether or not it ultimately became successful. Create a binary classifier model that will predict whether an applicant will become a successful business.


* Description of the stages of the deep learning process

    * Prepare the Data for Use on a Neural Network Model 

1. Read in the CSV file from the Resources folder into a Pandas DataFrame
2. Review the DataFrame, look for categorical variables that require encoding, as well as columns that could eventually define your features and target variables.
3. Encode the dataset’s categorical variables using OneHotEncoder, and then place the encoded variables into a new DataFrame.
4. Add the original DataFrame’s numerical variables to the DataFrame containing the encoded variables(use the pandas concat function).
5. Use the preprocessed data, create the features (X) and target (y) datasets. The target dataset should be defined by the preprocessed DataFrame column “IS_SUCCESSFUL”. 
6. Split the features and target sets into training and testing datasets.
7. Use scikit-learn's (StandardScaler) to scale the features data.

    * Compile and Evaluate a Binary Classification Model Using a Neural Network

1. Assign the number of input features, the number of layers, and the number of neurons on each layer using Tensorflow’s Keras
2. Compile and fit the model using the binary_crossentropy loss function, the adam optimizer, and the accuracy evaluation metric.
3. Evaluate the model using the test data to determine the model’s loss and accuracy.
4. Save and export your model as a HDF5 file under finished files, name the file AlphabetSoup.h5.

    * Optimize the Neural Network Model

1. Define at least two new deep neural network models
2. After finishing your models, display the accuracy scores achieved by each model, and compare the results. 
3. Save each of your models as an HDF5 file.

## Results
    *  Original model's results

Loss: 0.5537083148956299, Accuracy: 0.7309620976448059

    *   Alternative Model 1 results

Loss: 0.5619542002677917, Accuracy: 0.7297959327697754

    *   Alternative Model 2 Results

Loss: 0.7570863366127014, Accuracy: 0.5280466675758362
---

## Summary
Optimization of the model was tested by using different number of hidden layers for two of the models, and different epochs for two of the models.

    * Number of layers
The original model and Alternative one model use different number of hidden layers, with the original having 2 layers, while the alternative has one layer. With different number of hidden layers, the models reflect a very small difference in both their loss and accuarcy values, with loss being (o.5537 vs 0.5609) and accuracy being (0.7310 vs 0.7298). This is proof that adding more layers is not a guarantee for better performance, hence the reason why Trial and error is the only way to determine how “deep” a deep learning model should be.

    * Number of Epochs
The original model and Alternative two model use similar number of hidden layers(2), but employ different number of epochs with the original having 50 epochs, and Alt 2 model having 70 epochs. The models reflects a huge difference in both their loss (0.5537 vs 0.7571), and their accuracy (0.7310 vs 0.5280). It is clear the model’s training loss increases over the epochs as seen in Alt 2, where loss went up from 0.5537 to 0.7571, and  accuracy (for classification) decreased from 0.7310 to 0.5280.


## License
 The code is made without a license, however, the materials used for research are licensed.
---


