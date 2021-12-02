#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 20 18:02:22 2021
@author: aman
"""

# Importing required libraries
import numpy as np
import matplotlib.pyplot
import pandas as pd

# Importing dataaset
dataset = pd.read_csv('Data.csv')
# print(dataset)

# Defining Feature matrix(X) and target array(y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handling missing data
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
# Fitting the imputer object to our matrix of feature X. Here actually we are 
# not fitting imputer object to entire matrix of feature X. But we are fitting 
# it to a sub-matrix created with column2 and column3 as missing data are 
# there in these columns only. 
imp = imp.fit(X[:, 1:3])
# Here we are saying perform this imputation on column2(column index 1) and 
# column3(column index2). Since upper bound value is excluded, 
# so we have written 1:3 for column2 and column3.
 
# Above code only tells imputer that on which data you have to take the mean
# And it will not replace the missing data with the mean of the column
# We can achieve that with following line of code. 
X[:, 1:3] = imp.transform(X[:, 1:3])

# Encoding Categorical data. 
# ML modules works on mathematical calculations and hence we should provide it 
# with numerical values. So we should replace categorical data(i.e text values
# or strings) with a number like here we should replace - France with 0, 
# Germany with and Spain with 2 etc.(order not important here).We can achieve 
# that with following lines of code 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder() 
X[:, 0] = label_encoder.fit_transform(X[:, 0])
y = label_encoder.fit_transform(y)

"""
Now if you print X you will find Country name is replaced by some numerical 
values like - France with 0, Germany with 1, Spain with 2 etc.
"""
# print(X)
"""
But still there is some problem in it.Here ML model might understand that 
Spain have some kind of superiority over Germany and Germany have same over
France as Spain got the value 2 while Germany got 1 and France got the value 
0. So we should not stick to such encoding. The better option would be we 
replace country column with 3 columns each dedicated for 1 Country i.e 1 for 
France, 1 for Germany and 1 for Spain and assign boolean values to them.
For example - since First row country is France. So in first row we should 
assign 1 for France and 0 for Germany and Spain. We can achieve it with 
OneHotEncoder from sklearn.preprocessing library as follows(Imported the 
library above).
"""
# =============================================================================
# onehot_encoder = OneHotEncoder(sparse=False)
# country_encoded = onehot_encoder.fit_transform(X[:, 0].reshape(-1,1))
# X = np.concatenate((country_encoded, X[:, 1:3]), axis=1)
# print(X)
# =============================================================================
"""
We have achieved our desired goal as above but we can achieve the same with 
some other methods as well like.You can use any of these methods(i.e among
above method and following two methods) to one hot encode categorical data -
"""
### Method1 :-
# =============================================================================
# from sklearn.compose import ColumnTransformer
# 
# ct = ColumnTransformer(
#   [('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],  
#   remainder='passthrough'                                         
# )
# X = ct.fit_transform(X)
# print(X)
# =============================================================================
"""
Here [0] represents The column numbers to be transformed (here it is [0] but
there can be multiple columns as well like [0, 1, 3] etc.).
Here remainder='passthrough' represents - Leave the rest of the columns 
untouched(i.e leave all columns except the first column(i.e column whose index 
is 0) untouched) .
"""

### Method2 :-
a = np.array(X[:,0])
country_encoded = np.zeros((a.size, a.max()+1))
country_encoded[np.arange(a.size).astype(int), a.astype(int)] = 1
X = np.concatenate((country_encoded, X[:, 1:3]), axis=1)
print(X)

# Splitting data into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1)

# Feature scaling training and test data for better performance 
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
