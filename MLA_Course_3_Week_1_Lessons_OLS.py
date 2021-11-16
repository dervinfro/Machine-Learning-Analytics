import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

'''
************************
OLS and Machine Learning
************************
'''

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/sampledata.csv')



#set dataframe fields (value and medhinc) to log
#without researching as to why this was done in this weeks problem, I would venture to guess that the np.log was performed on the fields to normalize the data
df['log_value'] = np.log(df['value'])
df['log_medhinc'] = np.log(df['medhinc'])

#we need a constant variable in OLS models (SEE: https://www.theanalysisfactor.com/the-impact-of-removing-the-constant-from-a-regression-model-the-categorical-case/)
df['const'] = 1

X = df[['const','log_medhinc','walkscore','numreviews_med_5']]
y = df['log_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12) #setting the random state to a value ( it can be any value) will allow us to reproduce the split results each time we run the datasets.

w = np.dot(np.linalg.pinv(X_train), y_train)
print(w) #in the previous Statistics courses, we would refer to 'w' as beta

'''
After calculatinig our model's parameters using the training data, we could measure its performance on that set like we have done in the past. We might look at its r-squared, adjusted r-squared, and the root of the mean of the squared residuals (RMSE):
'''

n = len(y_train)
k = 3

training_predictions = np.dot(X_train, w) #the models predictions on the training data

SSR = sum((training_predictions - y_train)**2) #SSR = sum of square residuals
TSS = sum((y_train - np.mean(y_train))**2) #TSS = total sum of squares

r_squared = 1-(SSR)/(TSS)
adj_r_squared = 1 - ((n-1)/(n-k-1))*(1-r_squared)
RMSE = (SSR/n)**(1/2)
print('For the training data the r_squared is {}'.format(r_squared))
print('For the training data the adj. r_squared is {}'.format(adj_r_squared))
print('For the training data the RMSE is {}'.format(RMSE))
print('')

test_predictions = np.dot(X_test,w)
n = len(y_test)
SSR = sum((test_predictions-y_test)**2)
TSS = sum((y_test-np.mean(y_test))**2)

r_squared = 1-(SSR)/(TSS)
adj_r_squared = 1 - ((n-1)/(n-k-1))*(1-r_squared)
RMSE = (SSR/len(y_test))**(1/2)
print('For the test data the r_squared is {}'.format(r_squared))
print('For the test data the adj. r_squared is {}'.format(adj_r_squared))
print('For the test data the RMSE is {}'.format(RMSE))
