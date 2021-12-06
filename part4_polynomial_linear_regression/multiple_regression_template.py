#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 22:44:30 2021

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


# Step4 - Fitting Polynomial Linear Regression to the dataset
# Define your regressor here.With same name as used in step5 i.e 'regressor', 
# or change the name of regressor object in step5 also. 


# Step5 - Visualizing the Linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bulff (Regression model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Same as above but for smooth curve in the graph.
X_grid = np.arange(min(X), max(X), 0.1)
# Following line will just convert X_grid from array to matrix
X_grid = X_grid.reshape(len(X_grid), 1)
# Now again plot the graph same as above
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bulff (Regression model with smooth curve)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
