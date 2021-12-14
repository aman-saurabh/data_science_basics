#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 09:19:30 2021

@author: aman
"""

# Step1 - Importing required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Step2 - Importing dataaset
dataset = pd.read_csv('Mall_Customers.csv')

"""
In this datset client's whose spending score is more(i.e close to 100) is more 
likely to spend money in shopping in the mall and client's whose score is less 
are less likely to spend money in shopping in the mall. Here the objective is 
to group similar types of customers together in a group.But we don't know in 
which group and neither we know how many such groups we have.So this is 
typically a clustering problem and we will solve it using k-means clustering 
technique.
In the dataset many features are given but we will consider only annual income
and spending score as other features are not so important for deciding about 
the spending behavious of customers. 
"""
# Step3 - Preparing the dataset
X = dataset.iloc[:, [3,4]].values

# Step4 - Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, 
                    random_state=0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
plt.plot(range(1,11), wcss)
plt.title("The Elbow method")
plt.xlabel("Number of clusters")
plt.ylabel("WCSS")
plt.show()

# FRom graph we can see that optimal number of cluster = 5.Let's save this 
# value in a variable let's say n_optimal
n_optimal = 5

# Step5 - Applying k-means clustering to the mall dataset
kmeans = KMeans(n_clusters=n_optimal, init='k-means++', max_iter=300, 
                n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(X)
"""
dataset is clustered into 5 different groups.y_kmeans gives the cluster 
number for corresponding data-point in the dataset(or X).Also remember 
cluster number starts from 0 and not 1 i.e y_kmeans value for data-point 
which is assigned to first cluster is 0 and not 1.   

Above code will work fine for any clustering problem, you just have to change 
the dataset file name and columns of interest. But following visualizationis is
valid only for 2-dimensional space i.e if have used only two features of data 
for clustering.
"""
# Step6 - Visualising the cluster 
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', 
            label='Cluster1')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', 
            label='Cluster2')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', 
            label='Cluster3')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', 
            label='Cluster4')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', 
            label='Cluster5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='yellow', label='Centroids')
plt.title("Clusters of clients")
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()
"""
From graph we can see that Cluster1(coloured in red in graph) contains 
customers who have high income but they spend less.So we can call this 
cluster "Careful" inplace of "Cluster1". Similarly we can see that Cluster3 
contains customers who have high income and they spend also more.So they are 
the main targer customers and we can call this cluster as "Target".Similarly 
we can call cluster2 as "Standard"(as midium income, medium spending), 
cluster4 as "Careless"(as low income but high spending) and cluster5 as 
"Sensible"(as low income, low spending).So let's draw the graph again with 
new labels  
"""
plt.scatter(X[y_kmeans==0, 0], X[y_kmeans==0, 1], s=100, c='red', 
            label='Careful')
plt.scatter(X[y_kmeans==1, 0], X[y_kmeans==1, 1], s=100, c='blue', 
            label='Standard')
plt.scatter(X[y_kmeans==2, 0], X[y_kmeans==2, 1], s=100, c='green', 
            label='Target')
plt.scatter(X[y_kmeans==3, 0], X[y_kmeans==3, 1], s=100, c='cyan', 
            label='Careless')
plt.scatter(X[y_kmeans==4, 0], X[y_kmeans==4, 1], s=100, c='magenta', 
            label='Sensible')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
            s=200, c='yellow', label='Centroids')
plt.title("Clusters of clients")
plt.xlabel("Annual income(k$)")
plt.ylabel("Spending Score(1-100)")
plt.legend()
plt.show()