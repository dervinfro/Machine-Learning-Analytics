#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

pd.options.display.max_columns = None
pd.options.display.max_rows = None
'''
NOTE:  UPDATE....as of 18JAN this has been solved for question 4.
'''

df = pd.read_csv("/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/sampledata.csv")

df["log_value"] = np.log(df["value"])
df["log_medhinc"] = np.log(df["medhinc"])

X = df[["medhinc", "walkscore", "elem_score"]]
y = df["log_value"]
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=25) #setting the random_state for reproducibility 

#Note: this function assumes we're looking at these 3 specific features!
#As an exercise, you can try adapting it be more generic and work for any number of features
#X_train is the training data used to create the model
#test_data is any new data point that you want to create predictions for
def get_euclidean_distances (train_data, test_case):
	distances = []
	for i in range(len(train_data)):
		distance = ((train_data.iloc[i]["medhinc"]-test_case["medhinc"])**2 + (train_data.iloc[i]["walkscore"]-test_case["walkscore"])**2 + (train_data.iloc[i]["elem_score"]-test_case["elem_score"])**2)**(1/2)
		distances.append(distance)
	return (pd.Series(distances)) #Return the distances as a series

def find_nearest_neighbors (train_data, test_data, k):
	neighbors = []
	for i in range(len(test_data)):
		distances = get_euclidean_distances(train_data, test_data.iloc[i])
		distances = distances.sort_values()
		distances = distances[:k]
		distances = (1/distances)/sum(1/distances) # inversion
		distances = distances.replace(np.nan, 1) # drop all NaN that were created after the inversion - replace with 1
		neighbor_keys = distances # pass the pd.Series to the temp. variable
		neighbors.append(neighbor_keys) # neighbors list is appended with temp variable above
	return(pd.Series(neighbors)) #Returns the pd.Series


k = 6 #Initially, let's look at the 6 nearest neighbors
keys = find_nearest_neighbors(X_train, X_test, k) # the pd.Series return object assigned as "keys"

predictions = []
for key in keys: #For each 
	prediction = sum(key.values * y_train.iloc[key.index.values]) # weights (key.values) * y_train
	predictions.append(prediction)

n = len(y_test) #The number of observations
vars = 3 #The number of independent variables in the model
SSR = sum((predictions-y_test)**2)
TSS = sum((y_test-np.mean(y_test))**2)

r_squared = 1-(SSR)/(TSS)
RMSE = (SSR/len(y_test))**.5
adj_r_squared = adj_r_squared = 1-((n-1)/(n-vars-1))*(1-r_squared)
print("For a k of %i the r squared is %f" % (k,round(r_squared,3)) ) # correct answer for Q4 is: 0.223
print("For a k of %i the adjusted r squared is %f" % (k,adj_r_squared) )
print("For a k of %i the RMSE is %f" % (k,RMSE) )
'''
For a k of 6 the r squared is 0.223000
For a k of 6 the adjusted r squared is 0.172834
For a k of 6 the RMSE is 0.704573
'''