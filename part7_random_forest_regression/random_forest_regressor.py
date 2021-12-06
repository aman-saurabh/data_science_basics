#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 01:36:35 2021

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
# Commenting it since it may or may not be used based on the algorithms type.
"""
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y.reshape(-1, 1)) 
# Here just turning 1-d array to 2-d array with y.reshape(-1, 1) as 
# StandardScaler's transform method needs 2-d array.
"""

# Step5 - Fitting Random Forest Regression to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state=0)
# change value of 'n_estimators' for different values like 10, 100, 400 etc. 
# and check y_pred_value and graph for them
regressor.fit(X, y)
y_pred = regressor.predict(X)

# Step 6 - To find value of y for a particular value of X.
y_pred_value = regressor.predict(np.array([[6.5]]))

# Step7 - Visualizing the Random Forest regression result
X_grid = np.arange(min(X), max(X), 0.01)
# Following line will just convert X_grid from array to matrix
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or bluff (Random Forest Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
