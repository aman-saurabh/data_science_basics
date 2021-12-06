#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 11:12:59 2021

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
"""
Here we are considering only 2nd column in X (i.e only column at index 1)and 
only 4th(i.e last) column in y(i.e column at index 3 or -1). Which means in
both X and y we used only 1 column but we used different format in both cases 
why? Actually we could have defined X as "dataset.iloc[:, 1].values" also but
in that case it was treated as Array and not.
"""

# Step4 - Fitting Simple Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)
# predicting values of y corresponding to x
y_lin_reg_pred = lin_reg.predict(X)
# predicting salary with 6.5 years of experience i.e y_pred at X = 6.5
y_lin_reg_pred2 = lin_reg.predict(np.array([[6.5]]))

# Step5 - Fitting Polynomial Linear Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
"""
Now if you instect "X_poly" you will find it created 3 columns from 1 column 
of X and first column is the columns of 1.So in this case we don't need to 
add a columns of 1s manually like what we did in multiple linear regression
"""
poly_lin_reg = LinearRegression()
poly_lin_reg.fit(X_poly, y)
# predicting values of y corresponding to x
y_poly_reg_pred = poly_lin_reg.predict(X_poly)
# predicting salary with 6.5 years of experience i.e y_pred at X = 6.5
y_poly_reg_pred2 = poly_lin_reg.predict(poly_reg.fit_transform(np.array([[6.5]])))

# Visualizing the Linear regression result
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Truth or bluff (Linear Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

# Visualizing the polynomial regression result
plt.scatter(X, y, color='red')
plt.plot(X, poly_lin_reg.predict(poly_reg.fit_transform(X)), color='blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
"""
Now change the degree in Step5 and check graph for degree == 3,4,5 etc. You 
will find you get best fitted line with X = 5.
But there is one problem in the graph.If you see the graph you will find 
that.The line is nat a smooth curve. It seems like several small lines are 
connected together to make a big line.It is because it plotted graph by 
considering lines between points at interval of 1.We can get smooth curve if 
we consider higher resolution i.e instead by having predictions from 1-10 
incremented by 1, we should considerer predictions from 1-10 incremented by
small amount like by 0.1 or 0.01 etc.
"""
X_grid = np.arange(min(X), max(X), 0.1)
# Following line will just convert X_grid from array to matrix
X_grid = X_grid.reshape(len(X_grid), 1)
# Now again plot the graph same as above
plt.scatter(X, y, color='red')
plt.plot(X_grid, poly_lin_reg.predict(poly_reg.fit_transform(X_grid)), color='blue')
plt.title('Truth or bluff (Polynomial Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
