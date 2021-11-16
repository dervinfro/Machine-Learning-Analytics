#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 15:22:53 2021

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

df1 = df

# create dummy variable for df['genre]
genre_dummies = pd.get_dummies(df['genre'])

# Concat df and genre_dummies
df = pd.concat([df,genre_dummies], axis=1) #ensure the axis is set to 1, otherwise the dummies return as 'nan'

# Delete category column of df['genre']
df.drop(columns=['genre','company','country','director','name','released','gross','star','writer','year','rating','War'], inplace=True)


# Preprocess data by label encoding a categorical column.  
# With this LabelEncoder capability, I no longer need to use dummies (pd.get_dummies)
# labelencoder_X = LabelEncoder()
# df.loc[:, 'genre'] = labelencoder_X.fit_transform(df.loc[:,'genre'])

# set the dataframe to all movies with a budget greater than zero
df = df[df['budget'] > 0] #dataframe set to budget greater than zero

# Set the X variable to all necessary columns as well as the dummy values for df['genre']
X = df

# Scale data
# fit = compute mean and std to be used for scaling  
# transform = standardizing by centering and scaling
scaler = StandardScaler().fit(X) 
X_scaled = scaler.transform(X)


scaled_kmeans_score = []

# Question 4
for i in range(5,16):
    kmeans_model = KMeans(n_clusters = i, init="random", n_init=20, random_state=25)
    kmeans_model.fit(X_scaled)
    predictions = kmeans_model.predict(X_scaled)
    score = silhouette_score(X_scaled, kmeans_model.fit_predict(X_scaled))
    print(i, score)
    scaled_kmeans_score.append(score)

k=16

# Model fit and prediction in three lines
kmeans_model = KMeans(n_clusters = i, init="random", n_init=20, random_state=25)
kmeans_model.fit(X_scaled)
predictions = kmeans_model.predict(X_scaled)
 
 #Calculate the silhouette coefficients for each sample
sample_silhouette_coefficients = silhouette_samples(X_scaled, predictions)

cluster_averages = []
# Question 5
for i in range(5,k):
    # #First, get the coefficients for the ith cluster
    ith_cluster_coefficients = sample_silhouette_coefficients[predictions==i]
    
    # #Get the avereage coefficient for the cluster
    ith_cluster_coefficient_average = np.mean(ith_cluster_coefficients)
    
    # Append cluster averages
    cluster_averages.append(ith_cluster_coefficient_average)

low_score = np.min(cluster_averages) 

# Set the column 'cluster_kmeans' to the value of predictions
df['predictions'] = predictions

   

'''
# SEE VERSION 2 OF THIS CODE FOR A MUCH CLEANER WAY OF DATAFRAMES AND Q 6.


# QUESTION 6
# Convert predictions np.array into pd.Series to be able to concat
predictions_series = pd.Series(predictions)

# QUESTION 6 CONT.
# Convert sample_sil_coef into pd.Series to be able to concat
sample_silh_coef_series = pd.Series(sample_silhouette_coefficients)

# QUESTION 6 CONT.
# Reset index from previous budget > 0 drop
df = df.reset_index(inplace=False, drop=True)

# QUESTION 6 CONT.
# Concat df, predictions_series and sample_silh_coef_series
df = pd.concat([df,predictions_series.rename('predictions'), sample_silh_coef_series.rename('sil_coef')], axis=1)

# QUESTION 6 CONT.
# List all of the dataframe for cluster 5 (Horror)
predictions_var = df[df['predictions'] == 5].sort_values('score')

# QUESTION 6 CONT.
# predictions_var = LEFT
# df.iloc[:,[6,12]] = RIGHT.  Specifically two columns: name and votes
# Pandas merge two dataframes to bring in movie titles ('name')
# NOTE: the merge fields 'left_on' and 'right_on' MUST be included in the merge fields.
df3 = pd.merge(predictions_var, df1.iloc[:,[6,12]], how='left', left_on=['votes'], right_on=['votes'])

# QUESTION 6 CONT.
# Return row with the max score
question_6 = df3[df3['score'] == df3['score'].max()]

# QUESTION 6 CONT.
# Return row with the two fields of: score & name
question_6_ver2 = question_6[['score','name']]

# QUESTION 6 CONT.
# Show all rows for the Horror genre and sort the scores
score_var = df[df['Horror'] == 1].sort_values('score')
'''
       


components = 17

gmm = GaussianMixture(n_components=components, covariance_type='diag', random_state=25)
gmm.fit(X_scaled)
die_hard_record = [28000000.0,132,8.2,763000,1,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0,	0]

# This die_hard_record is scaled and fit from X......then it is transformed.
dhr_scaled = scaler.transform([die_hard_record])

# dhr_scaled is predicted and probability
print(gmm.predict_proba(dhr_scaled))



    