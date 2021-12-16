#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 16 22:41:23 2021

@author: aman
"""

# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step2 - Importing dataaset
dataset = pd.read_csv('Market_Basket_Optimisation.csv', header=None)
"""
Here we have mentioned header=None as dataset doesn't contain header. The 
first line (shrimp, almonds, avocado...) represents data and not the header.
The total data in the dataset = 7501 and not 7500.
-----------------------------------------------------------------------------
The dataset that is provided here simply contains the transactions details i.e
only information about which transaction contains which items. But all items 
in the transaction is contained here in a seperate column which is not a 
proper format for input data in most machine learning models.So we need to 
restructure it.  
""" 
# Step3 - Restructuring data
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i,j]) for j in range(0, 20)])
    
# Step4 - Importing and applying apriori function
from apriori import apriori
# Here we are not importing apriori function from any library but we are 
# importing it from a local file apriori.py which we have stored in the same 
# location where current file is stored.
rules = apriori(transactions= transactions, min_support=0.003, min_confidence=0.2, min_lift=3, 
                min_length=2) 
"""
min_length -> Minimun number of items in a rule
min_support -> Minimum frequency of purchase of item in a day -> In this case 
we are considering minimum 3 times a day i.e minimum support = 3*7/7500 
= 0.0028 = 0.003. If it doesn't work we will change it to 0.004 or 0.005 and 
try again.
min_confidence -> Since it is a large dataset containing many items so chances 
of high confidence(like 0.8 or 0.75) for any antecedent and consequent is very 
less.So we will kept it 0.2 first and try.If it doesn't work we will change it 
to 0.3, 0.4 etc. 
min_lift -> lift less than 1 tells that association between antecedent and 
consequent is not good.So we should always choose lift > 1.Here dataset is 
large so let's start with 3, if it doesn't work, we will try it with 4,5,6 etc.      
"""

# Step5 - Visualizing the result
results = list(rules)
list_results = [list(results[i][0]) for i in range(0,len(results))]
