#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 09:06:11 2021

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

# Step5 - Fitting Decision Tree Regression to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)
y_pred = regressor.predict(X)

# Step 6 - To find value of y for a particular value of X.
y_pred_value = regressor.predict(np.array([[6.5]]))

# Step7 - Visualizing the Decision Tree Regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bluff (Decision Tree Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

"""
But if you see the graph you will find the value is increasing from one 
point to another in an interval and the graph is continious .But we have 
learned that Decision tree predicts same value for an entire interval 
(that is average of all values in that interval) and its graph is 
non-continious. So is there anything wrong in this model.No actually we 
need to display the graph in higher resolution in order to get the graph in 
desired format.
"""
X_grid = np.arange(min(X), max(X), 0.01)
# Following line will just convert X_grid from array to matrix
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or bluff (Decision Tree Regression Model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()