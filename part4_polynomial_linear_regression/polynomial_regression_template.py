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

# Step5 - Fitting Polynomial Linear Regression to the dataset
# Define your regressor here.With same name as used in step5 i.e 'regressor', 
# or change the name of regressor object in step5 also. 

# Step 6 - To find value of y for a particular value of X.
# Case 1 - If values(i.e X and y) are not featured scaled explicitely
y_pred_value = regressor.predict(np.array([[6.5]]))

# Case 2 - If values are feature scaled explicitely
"""
y_pred_value = regressor.predict(sc_X.transform(np.array([[6.5]])))
# But above line will give you result in scaled format. So execute following 
# code to get actual value of y corresponding to X=6.5.
y_pred_value = sc_y.inverse_transform(y_pred2)
"""

# Step7 - Visualizing the Linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, regressor.predict(X), color='blue')
plt.title('Truth or bluff (Regression model)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Same as above but for smooth curve in the graph.
X_grid = np.arange(min(X), max(X), 0.1)
# Following line will just convert X_grid from array to matrix
X_grid = X_grid.reshape(len(X_grid), 1)
# Now again plot the graph same as above
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Truth or bluff (Regression model with smooth curve)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
