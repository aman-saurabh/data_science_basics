#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  1 22:48:34 2021

@author: aman
"""

# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step2 - Importing dataaset
dataset = pd.read_csv('Salary_Data.csv')

# Step3 - Defining Feature matrix(X) and target array(y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Step4, Step5 and Step6 is not required as there is no missing data in the 
#dataset 

# Step7 - Splitting data into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, 
                                                    random_state=0)

# Step8 is also not required as some libraries internally takes care of 
# feature Scaling.And Simple linear Regression is one of them.

# Step9 - Fitting Simple Linear Regression to the training data. 
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Step10 - Predicting the test data result
y_pred = regressor.predict(X_test)

# Step11 - Visualising the training set result
plt.scatter(X_train, y_train, color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('Salary Vs Experience(Training Set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# Step12 - Visualising the test set result
plt.scatter(X_test, y_test, color='red')
plt.plot(X_test, regressor.predict(X_test), color='blue')
plt.title('Salary Vs Experience(Test Set)')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()