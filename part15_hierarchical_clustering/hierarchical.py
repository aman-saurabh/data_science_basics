#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 15 00:08:16 2021

@author: aman
"""
# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step2 - Importing dataaset
dataset = pd.read_csv('Mall_Customers.csv')

# Step3 - Preparing the dataset
X = dataset.iloc[:, [3,4]].values

# Step4 - Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("Dendrogram")
plt.xlabel("Customers")
plt.ylabel("Euclidean Distance")
plt.show()

# FRom graph we can see that optimal number of cluster = 5.Let's save this 
# value in a variable let's say n_optimal
n_optimal = 5

# Step5 - Applying hierarchical clustering to the mall dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters=n_optimal, affinity='euclidean', 
                             linkage='ward')
y_hc = hc.fit_predict(X)

# Step6 - Visualising the cluster 
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', 
            label='Cluster1')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', 
            label='Cluster2')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', 
            label='Cluster3')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', 
            label='Cluster4')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', 
            label='Cluster5')
# Centroids are not as necessary in hierarchical clustering as it was there in
# k_means clustering. 
plt.title("Clusters of clients")
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()

# Renaming clusters names with some meaningful values.Check k-means clustering 
# file to learn about it in more details
plt.scatter(X[y_hc==0, 0], X[y_hc==0, 1], s=100, c='red', 
            label='Careful')
plt.scatter(X[y_hc==1, 0], X[y_hc==1, 1], s=100, c='blue', 
            label='Standard')
plt.scatter(X[y_hc==2, 0], X[y_hc==2, 1], s=100, c='green', 
            label='Target')
plt.scatter(X[y_hc==3, 0], X[y_hc==3, 1], s=100, c='cyan', 
            label='Careless')
plt.scatter(X[y_hc==4, 0], X[y_hc==4, 1], s=100, c='magenta', 
            label='Sensible')
plt.title("Clusters of clients")
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()