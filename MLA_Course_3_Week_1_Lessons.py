import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/sampledata.csv')

'''
**************************
K-Nearest Neighbors (K-NN)
**************************
'''

#set dataframe fields (value and medhinc) to log
#without researching as to why this was done in this weeks problem, I would venture to guess that the np.log was performed on the fields to normalize the data
df['log_value'] = np.log(df['value'])
df['log_medhinc'] = np.log(df['medhinc'])


#set X to the dataframe with (log_medhinc, walkscore, numreviews_med_5)
X = df[['log_medhinc','walkscore','numreviews_med_5']]
y = df['log_value']

#the test_train_split comes from the sklearn model_selection
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=12) #setting the random state to a value ( it can be any value) will allow us to reproduce the split results each time we run the datasets.

'''
Next, we'll create a simple function to calculate Euclidean distance. If you pass it a data point, it iterates through the training data and calculates how far the new data is from each observation in the training data:
'''
#this function is designed around using THREE features.  
def get_euclidean_distances(train_data, test_case):
    distances = []
    for i in range(len(train_data)):
        distance = ((train_data.iloc[i]['log_medhinc']-test_case['log_medhinc'])**2 + (train_data.iloc[i]['walkscore']-test_case['walkscore'])**2 + (train_data.iloc[i]['numreviews_med_5']-test_case['numreviews_med_5'])**2)**(1/2)
        distances.append(distance)
    return (pd.Series(distances))

'''
Next, we'll write a function that uses get_euclidean_distances() to find the k nearest neighbors. It takes the training data, the new data to be considered, and k. For each row in the new data, it finds the distances from each row in the training data. Then, it sorts those distances to find the k training observations that are closest. Finally, it returns the indices for those k closest observations.
'''
def find_nearest_neighbors(train_data, test_data, k):
    neighbors=[]
    for i in range(len(test_data)):
        distances = get_euclidean_distances(X_train, test_data.iloc[i])
        distances = distances.sort_values()
        neighbor_keys = distances[:k].index.values #find the indices of the K nearest neighbors
        neighbors.append(neighbor_keys)
    return(neighbors) #return the indices of the K nearest neighbors

'''
Finally, we'll use these two functions to find the k nearest neighbors in the training set for the test set. We'll find out the values for log_value for those nearest neighbors, and our prediction will be their mean.
'''
#k = 2 #initially we'll look at the five (5) closes neighbors
for k in range(2,21):
    keys = find_nearest_neighbors(X_train, X_test, k)
    predictions = []
    for key in keys:
        prediction = y_train.iloc[key].mean()
        predictions.append(prediction)
        
    
    n = len(y_test)
    vars = 3 #the number of independent variables in the model
    SSR = sum((predictions-y_test)**2)
    TSS = sum((y_test-np.mean(y_test))**2)
    
    r_squared = 1-(SSR)/(TSS)
    RMSE = (SSR/len(y_test))**.5
    adj_r_squared = 1-((n-1)/(n-vars-1))*(1-r_squared)
    #print('For a k of {} the r_squared is {}'.format(k,r_squared))
    #print('For a k of {} the adj. r_squared is {}'.format(k, adj_r_squared))
    print('For a k of {} the RMSE is {}'.format(k,RMSE))

