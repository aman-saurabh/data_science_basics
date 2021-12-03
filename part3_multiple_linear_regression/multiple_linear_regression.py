#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  3 09:42:07 2021

@author: aman
"""

# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step2 - Importing dataaset
dataset = pd.read_csv('50_Startups.csv')

# Step3 - Defining Feature matrix(X) and target array(y)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Step4 - Encoding Categorical data. 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label_encoder = LabelEncoder() 
X[:, 3] = label_encoder.fit_transform(X[:, 3])

# Step5 - One hot encoding previously encoded categorical data
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
   [('my_encoder', OneHotEncoder(categories='auto'), [3])],  
   remainder='passthrough'                                         
)
X = ct.fit_transform(X)
# State column was on index 3 but the 3 columns which were created in previous 
# step was created on index 0, 1 and 2 i.e index of all independent columns 
# changed.That isn't a problem.

# Step6 - Avoiding Dummy variable trap
X = X[:, 1:]
# However Python libraries for linear regression takes care of Dummy variable 
# trap for us.So we don't need above line of code.The purpose was to show how 
# to achieve it.
   
# Step7 - Splitting data into training data and test data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=1)

# Step8 - Fitting Multiple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
 
# Step9 - Predicting the test set result
y_pred = regressor.predict(X_test)

# ***Important - Building the optimal model using Backward elimination - 
"""
Above model is built using "All-in" method which is not an optimal model.
So now we will build an optimal model using Backward elimination method.

Formula for Multiple linear regression :
    y = b0 + b1*x1 + b2*x2 + ... + bn*xn
We can rewrite it as :
    y = b0*1 + b1*x1 + b2*x2 + ... + bn*xn
or,
    y = b0*x0 + b1*x1 + b2*x2 + ... + bn*xn where x0 = 1
    
Most libraries like what we use above (i.e LinearRegression library) takes 
care of y-intercept b0. But some libraries like what we are going to use next
(i.e statsmodel library) do not. So in such cases we need to take care of b0.
We can achieve that by adding an extra column of dummy variable(of value - 1).
In that case we can apply Multiple linear regression as :  
    y = b0*x0 + b1*x1 + b2*x2 + ... + bn*xn where x0 = 1
"""
import statsmodels.api as sm
# creating an extra column of dummy variable of value 1 at the beginning i.e 
# at index 0 of X
X = np.append(arr=np.ones((50, 1)).astype(float), values=X, axis=1)
X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
"""
Check following section of the result :
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const       5.013e+04   6884.820      7.281      0.000    3.62e+04     6.4e+04
x1           198.7888   3371.007      0.059      0.953   -6595.030    6992.607
x2           -41.8870   3256.039     -0.013      0.990   -6604.003    6520.229
x3             0.8060      0.046     17.369      0.000       0.712       0.900
x4            -0.0270      0.052     -0.517      0.608      -0.132       0.078
x5             0.0270      0.017      1.574      0.123      -0.008       0.062
==============================================================================

Here x1,x2,x3 ... represents independent variabes and "P>|t|" represents the 
p-value.We should eliminate the variable with hightest p-value(i.e X2 in this 
case). We will repeat the process and eliminate variables one by one until
the highest p-value becomes less than our set "Significance level" i.e 0.05 in 
this case 
"""
X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
"""
Now you will see p-value for x1(which represents index 3 now) is 0.000 and 
p-value for x2(which represents index 5 now) is 0.060.Since our set 
"Significance level" i.e 0.05. So we should remove x2 also but since it is too 
close to significance level.So we should consider some other creterion also to
decide whether we should keep this variable or not, like R-squared and 
Adjusted R-squared. We will learn about these creterions later.
"""
X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
regressor_OLS.summary()
"""
Now it is clear that R&D spend has very great impact on profit, while 
marketing spent has little and other creterion like Administration and State 
has almost no impact on profit. So now we can create the Multiple linear 
regression model again as above by considering only R&D column as independent 
variable and ignoring all other columns.  
"""
