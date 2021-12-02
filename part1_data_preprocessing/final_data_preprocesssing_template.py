#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 22:29:34 2021

@author: aman
This file is just a clean copy of "data_preprocessing_template.py" file.
Created this clean template to copy and use its code in other files as almost 
all templates will have requirement of these preprocessing steps.
"""


# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step2 - Importing dataaset
dataset = pd.read_csv('Data.csv')

# Step3 - Defining Feature matrix(X) and target array(y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Step4 - Handling missing data
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
imp = imp.fit(X[:, 1:3])
X[:, 1:3] = imp.transform(X[:, 1:3])

# Step5 - Encoding Categorical data. 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder() 
X[:, 0] = label_encoder.fit_transform(X[:, 0])
y = label_encoder.fit_transform(y)

# Step6 - One hot encoding previously encoded categorical data
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
   [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],  
   remainder='passthrough'                                         
)
X = ct.fit_transform(X)

# Step7 - Splitting data into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1)

# Step8 - Feature scaling training and test data for better performance 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
