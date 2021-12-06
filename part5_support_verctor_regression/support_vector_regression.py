#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 22:54:16 2021

@author: aman
"""

# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step2 - Importing dataaset
dataset = pd.read_csv('Position_Salaries.csv')

# Step3 - Defining Feature matrix(X) and target array(y)
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, -1].values

# Step4 - Feature Scaling
# Since SVR is not very commanly used machine learning algorithm especially in 
# regression. So feature scaling is not performed implicitely by the model and 
# we users needs to perform it explicitely.   
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
# Here turning 1-d array to 2-d array.
y = sc_y.fit_transform(y.reshape(-1, 1)) 

# Step4 - Fitting Support vector Regression to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel='rbf')
regressor.fit(X, y)
# predicting values of y corresponding to x
y_pred = regressor.predict(X)
# predicting salary with 6.5 years of experience i.e y_pred at X = 6.5
y_pred2 = regressor.predict(sc_X.transform(np.array([[6.5]])))
# But above line will give you result in scaled format.So execute following 
# code to get actual value of y corresponding to X=6.5
y_pred_final = sc_y.inverse_transform(y_pred2)

# Step5 - Visualizing the support vector regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bulff (Regression model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()