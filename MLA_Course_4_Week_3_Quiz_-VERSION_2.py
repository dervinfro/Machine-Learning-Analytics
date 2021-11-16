#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 10:45:50 2021

@author: user
"""

import numpy as np 
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 4/movieclustering.csv')

# Create list of variables to drop
drop = ['company','country','director','name','released','gross','star','writer','year','rating']

# Drop any rows with budget equal to or less than zero
df = df[df['budget'] > 0]

# Reset the index.  
# The drop=True will prevent an index column from being added to the dataframe.
df = df.reset_index(drop=True)

# Set "variables" variable to the dataframe after dropping the columns in the list
variables = df.drop(drop, axis=1)

# Get the dummies from the "genre" column
X = pd.get_dummies(variables, prefix='', prefix_sep='', columns=['genre'])

scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)

'''
scaled_kmeans_score = []

# Question 4
for i in range(5,16):
    kmeans_model = KMeans(n_clusters = i, init="random", n_init=20, random_state=25)
    kmeans_model.fit(X_scaled)
    predictions = kmeans_model.predict(X_scaled)
    score = silhouette_score(X_scaled, kmeans_model.fit_predict(X_scaled))
    print(i, score)
    scaled_kmeans_score.append(score)
 '''   
    
k=13

# Model fit and prediction in three lines
kmeans_model = KMeans(n_clusters = k, init="random", n_init=20, random_state=25)
kmeans_model.fit(X_scaled)
predictions = kmeans_model.predict(X_scaled)
 
 #Calculate the silhouette coefficients for each sample
sample_silhouette_coefficients = silhouette_samples(X_scaled, predictions)

cluster_averages = []

# Question 5
for i in range(0,k):
    # #First, get the coefficients for the ith cluster
    ith_cluster_coefficients = sample_silhouette_coefficients[predictions==i]
    
    # #Get the avereage coefficient for the cluster
    ith_cluster_coefficient_average = np.mean(ith_cluster_coefficients)
    
    # Append cluster averages
    cluster_averages.append(ith_cluster_coefficient_average)

low_score = np.min(cluster_averages)

# QUESTION 6
# Add the predictions score the to the dataframe as a new column
df['predictions'] = predictions

# Add the sample_silhouette_coefficients score to the dataframe as a new column
df['sample_silhouette_coefficients'] = sample_silhouette_coefficients

# Set the q6 variable to show all rows of the genre 'Horror'
q6 = df[df['genre'] == 'Horror']

# Set the q6 variable to show the max score of the genre 'Horror'
q6 = q6[q6.score == q6.score.max()]

