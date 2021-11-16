import pandas as pd
import numpy as np
import statistics

from sklearn.model_selection import train_test_split

pd.options.mode.chained_assignment = None #default value is "warn".  NOTE: this line of code is here due to SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame
#SEE:https://stackoverflow.com/questions/20625582/how-to-deal-with-settingwithcopywarning-in-pandas

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/sampledata.csv')

df['log_value'] = np.log(df['value'])
df['const'] = 1


############
#QUESTION 1#
############
'''
As usual, the target variable will be log_value. The independent variables will be medhinc (median neighborhood income), elem_score (neighborhood elementary score according to Greatschools.com), and walkscore (according to walkscore.com).

Start by creating the log_value variable, and then divding your data into a training and test set; do a 75/25 split, and use random_state=25 so we all get the same results.

Create a linear regression using OLS. What R-squared do you get when you evaluate your model on the training data? Round to three decimal places.
'''

    
    
X = df[['const','medhinc','elem_score','walkscore']]
y = df['log_value']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

w = np.dot(np.linalg.pinv(X_train), y_train)
#print(w)

n = len(y_train)
k = 3 # number of X (independent) variables ** DOES NOT INCLUDE const **

training_predictions = np.dot(X_train, w) #the models predictions on the training data

SSR = sum((training_predictions - y_train)**2) #SSR = sum of square residuals
TSS = sum((y_train - np.mean(y_train))**2) #TSS = total sum of squares

r_squared = 1-(SSR)/(TSS)
adj_r_squared = 1 - ((n-1)/(n-k-1))*(1-r_squared)
RMSE = (SSR/n)**(1/2)
print('Q1 - For the training data the r_squared is {}'.format(round(r_squared,3)))
 

############
#QUESTION 2#
############
'''
Now try your model on the test set. What R-squared do you calculate for the test data? Round to three decimal places.
'''

test_predictions = np.dot(X_test,w)
n_question2 = len(y_test)
SSR_question2 = sum((test_predictions-y_test)**2)
TSS_question2 = sum((y_test-np.mean(y_test))**2)

r_squared_question2 = 1-(SSR_question2)/(TSS_question2)

print('Q2 - For the test data the r_squared is {}'.format(round(r_squared_question2,3)))


############
#QUESTION 3#
############
'''
Now, create a simple k-NN model using Euclidean distance to calculate closeness. Use k=6 and evaluate it on the test set.

What R-squared do you get?
'''

def Question_3():
    #this function is designed around using THREE features (medhinc, elem_score and walkscore).  
    def get_euclidean_distances(train_data, test_case):
        distances = []
        for i in range(len(train_data)):
            distance = ((train_data.iloc[i]['medhinc']-test_case['medhinc'])**2 + 
                        (train_data.iloc[i]['elem_score']-test_case['elem_score'])**2 + 
                        (train_data.iloc[i]['walkscore']-test_case['walkscore'])**2)**(1/2)
            distances.append(distance)
        return (pd.Series(distances))
    
    '''
    Next, we'll write a function that uses get_euclidean_distances() to find the k nearest neighbors. It takes the training data, the new data to be considered, and k. For each row in the new data, it finds the distances from each row in the training data. Then, it sorts those distances to find the k training observations that are closest. Finally, it returns the indices for those k closest observations.
    '''
    def find_nearest_neighbors_Q3(train_data, test_data, k):
        neighbors=[]
        for i in range(len(test_data)):
            distances = get_euclidean_distances(X_train, test_data.iloc[i])
            distances = distances.sort_values()
            distances = distances[:k] # this line sets the distances list of get_euclidean_distances method to the first K fields in the sorted list.  In this case K is equal to (SEE: the k value below)
            neighbor_keys = distances.index.values #find the indices of the K nearest neighbors
            neighbors.append(neighbor_keys)
        return(neighbors) #return the indices of the K nearest neighbors
    
    '''
    Finally, we'll use these two functions to find the k nearest neighbors in the training set for the test set. We'll find out the values for log_value for those nearest neighbors, and our prediction will be their mean.
    '''
    k = 6 #initially we'll look at the six (6) closes neighbors
    
    keys = find_nearest_neighbors_Q3(X_train, X_test, k)
    predictions_Q3 = []
    for key in keys:
        prediction = y_train.iloc[key].mean()
        predictions_Q3.append(prediction)
        
    
    n_question3 = len(y_test)
    vars = 3 #the number of independent variables in the model
    SSR_question3 = sum((predictions_Q3-y_test)**2)
    TSS_question3 = sum((y_test-np.mean(y_test))**2)
    
    r_squared_question3 = 1-(SSR_question3)/(TSS_question3)
    print('Q3 - The r_squared is {}'.format(round(r_squared_question3, 3)))
Question_3()

############
#QUESTION 4#
############
'''
For these purposes, use the following procedure:

1. After finding the k nearest neighbors, take the inverse of their distances and add them together.

2. For each neighbor, divide the inverse of its distance by the sum of the inverses. That number will be the weight given to that neighbor. For example, if there are three neighbors, with distances 1, 2, 3, the sum of the inverses is approximately 1.833. The weight for the first neighbor will be .546 (1/1/1.833); for the second, .273 (1/2/1.833); for the third, .182 (1/3/1.833).

3. Use those weights to calculate the prediction from the target variables in the trainng set.

Again creating a model using k=6, calculate the R-squared for the test set using Euclidean distance, but this time weight the neighbors by distance using the method described above.

What is the R-squared for the test set?
'''


def Question_4():
    
    #Note: this function assumes we're looking at these 3 specific features!
    #As an exercise, you can try adapting it be more generic and work for any number of features
    #X_train is the training data used to create the model
    #test_data is any new data point that you want to create predictions for
    def get_euclidean_distances (train_data, test_case):
        distances = []
        for i in range(len(train_data)):
            distance = ((train_data.iloc[i]["medhinc"]-test_case["medhinc"])**2 + 
                        (train_data.iloc[i]["walkscore"]-test_case["walkscore"])**2 + 
                        (train_data.iloc[i]["elem_score"]-test_case["elem_score"])**2)**(1/2)
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
        prediction = sum(key.values * y_train.iloc[key.index.values]) # SUM OF (weights (AKA key.values) * y_train)
        predictions.append(prediction)
    
    n = len(y_test) #The number of observations
    vars = 3 #The number of independent variables in the model
    SSR = sum((predictions-y_test)**2)
    TSS = sum((y_test-np.mean(y_test))**2)
    
    r_squared = 1-(SSR)/(TSS)
    RMSE = (SSR/len(y_test))**.5
    adj_r_squared = adj_r_squared = 1-((n-1)/(n-vars-1))*(1-r_squared)
    print('Q4 - The r_squared is {}'.format(round(r_squared, 3)))
    #print("For a k of %i the r squared is %f" % (k,round(r_squared,3)) ) # correct answer for Q4 is: 0.223
    #print("For a k of %i the adjusted r squared is %f" % (k,adj_r_squared) )
    #print("For a k of %i the RMSE is %f" % (k,RMSE) )

Question_4()

############
#QUESTION 5#
############
'''
Another method would be to use Manhattan distance--that is, ∑i=0n∣pi−qi∣ \sum_{i=0}^{n} {|p_i-q_i|} ∑i=0n​∣pi​−qi​∣, where p are the training data, q are the test data, i is the index of the feature being considered, and n are the number of features.

Create a model using k=6 with Manhattan distance, without weighting. What R-squared do you calculate when you use your model with the test set? Round to three decimal places.
'''
def Question_5():
    
    def get_manhattan_distances(train_data, test_data):
        distances = []
        for i in range(len(train_data)):
            distance = (np.abs(train_data.iloc[i]['medhinc'] - test_data['medhinc']) + 
                        np.abs(train_data.iloc[i]['elem_score'] - test_data['elem_score']) +
                        np.abs(train_data.iloc[i]['walkscore'] - test_data['walkscore']))
            distances.append(distance)
        return (pd.Series(distances))
    
    '''
    Next, we'll write a function that uses get_euclidean_distances() to find the k nearest neighbors. It takes the training data, the new data to be considered, and k. For each row in the new data, it finds the distances from each row in the training data. Then, it sorts those distances to find the k training observations that are closest. Finally, it returns the indices for those k closest observations.
    '''
    def find_nearest_neighbors_Q5(train_data, test_data, k):
        neighbors=[]
        for i in range(len(test_data)):
            distances = get_manhattan_distances(train_data, test_data.iloc[i])
            distances = distances.sort_values()
            distances = distances[:k] # this line sets the distances list of get_euclidean_distances method to the first K fields in the sorted list.  In this case K is equal to (SEE: the k value below)
            neighbor_keys = distances.index.values #find the indices of the K nearest neighbors
            neighbors.append(neighbor_keys)
        return(neighbors) #return the indices of the K nearest neighbors    
            
    k = 6 #initially we'll look at the six (6) closes neighbors
    
    keys = find_nearest_neighbors_Q5(X_train, X_test, k)
    predictions_Q5 = []
    for key in keys:
        prediction = y_train.iloc[key].mean()
        predictions_Q5.append(prediction)
    
        
    n_question5 = len(y_test)
    vars = 3 #the number of independent variables in the model
    SSR_question5 = sum((predictions_Q5-y_test)**2)
    TSS_question5 = sum((y_test-np.mean(y_test))**2)
    
    r_squared_question5 = 1-(SSR_question5)/(TSS_question5)
    print('Q5 - The r_squared is {}'.format(round(r_squared_question5, 3)))
Question_5()
            
    
    
############
#QUESTION 6#
############
'''
Lastly, calculate r-squared on the test set using Manhattan distance, and this time weighting by distance. What is the r-squared? Round to three decimal places.
'''

def Question_6():
    
    def get_manhattan_distances(train_data, test_data):
        distances = []
        for i in range(len(train_data)):
            distance = (np.abs(train_data.iloc[i]['medhinc'] - test_data['medhinc']) + 
                        np.abs(train_data.iloc[i]['elem_score'] - test_data['elem_score']) +
                        np.abs(train_data.iloc[i]['walkscore'] - test_data['walkscore']))
            distances.append(distance)
        return (pd.Series(distances))    
    
    '''
    Next, we'll write a function that uses get_euclidean_distances() to find the k nearest neighbors. It takes the training data, the new data to be considered, and k. For each row in the new data, it finds the distances from each row in the training data. Then, it sorts those distances to find the k training observations that are closest. Finally, it returns the indices for those k closest observations.
    '''
    def find_nearest_neighbors_Q6(train_data, test_data, k):
        neighbors=[]
        for i in range(len(test_data)):
            distances = get_manhattan_distances(train_data, test_data.iloc[i])
            distances = distances.sort_values()
            distances = distances[:k] # this line sets the distances list of get_euclidean_distances method to the first K fields in the sorted list.  In this case K is equal to (SEE: the k value below)
            distances = (1/distances)/sum(1/distances)
            distances = distances.replace(np.nan, 1)
            neighbor_keys = distances #find the indices of the K nearest neighbors
            neighbors.append(neighbor_keys)
        return(pd.Series(neighbors)) #return the indices of the K nearest neighbors 
    
    k = 6 #initially we'll look at the six (6) closes neighbors
    
    keys = find_nearest_neighbors_Q6(X_train, X_test, k)
    
    predictions_Q6 = []
    for key in keys:
        prediction = sum(key.values * y_train.iloc[key.index.values]) # SUM OF (weights (AKA key.values) * y_train)
        predictions_Q6.append(prediction)
    
        
    n_question6 = len(y_test)
    vars = 3 #the number of independent variables in the model
    SSR_question6 = sum((predictions_Q6-y_test)**2)
    TSS_question6 = sum((y_test-np.mean(y_test))**2)
    
    r_squared_question6 = 1-(SSR_question6)/(TSS_question6)
    print('Q6 - The r_squared is {}'.format(round(r_squared_question6, 3)))    

Question_6()


###################
#QUESTION 7, 8 & 9#
###################
'''
There are different ways to handle this. We might work with the log of medhinc, as we have sometimes done already, or use min-max scaling. Another approach is z normalization. We saw how to do this in module 1 of the program:
z=xi−xˉS z =  \frac{x_i - \bar{x}}{S} z=Sxi​−xˉ​

Where xˉ \bar{x} xˉ is the mean and S is the sample standard deviation.

We'll learn more about normalization and standardization throughout the course. For now, follow this procedure:

1. Divide the data into training and test sets, using the a 75/25 split and random_state=25.

2. Find the training mean and standard deviation for each independent variable.

3. Normalize the independent variables in both the training and test sets with the mean and standard deviations from the training set.

After normalization, what is the best value for k that you can find for the distance-weighted Euclidean model?
'''

def Question_7():
    
    def get_manhattan_distances(train_data, test_data):
        distances = []
        for i in range(len(train_data)):
            distance = (np.abs(train_data.iloc[i]['medhinc'] - test_data['medhinc']) + 
                        np.abs(train_data.iloc[i]['elem_score'] - test_data['elem_score']) +
                        np.abs(train_data.iloc[i]['walkscore'] - test_data['walkscore']))
            distances.append(distance)
        return (pd.Series(distances))    


    def find_nearest_neighbors_Q7(train_data, test_data, k):
        neighbors=[]
        for i in range(len(test_data)):
            distances = get_manhattan_distances(train_data, test_data.iloc[i])
            distances = distances.sort_values()
            distances = distances[:k] # this line sets the distances list of get_euclidean_distances method to the first K fields in the sorted list.  In this case K is equal to (SEE: the k value below)
            distances = (1/distances)/sum(1/distances)
            distances = distances.replace(np.nan, 1)
            neighbor_keys = distances #find the indices of the K nearest neighbors
            neighbors.append(neighbor_keys)
        return(pd.Series(neighbors)) #return the indices of the K nearest neighbors     
    
    '''
    NOTE: As of 20 JAN, Q7 is now reporting the correct answer.  The issue here was that I was normalizing the data PRIOR to determining the distance and the neighbors.  That was incorrect.  I need to run the distance and neighbors AND THEN normalize the data for predictions.
    '''
    medmean = X_train["medhinc"].mean()
    walmean = X_train["walkscore"].mean()
    elemean = X_train["elem_score"].mean()
    medstd = X_train["medhinc"].std()
    walstd = X_train["walkscore"].std()
    elestd = X_train["elem_score"].std()
    X_train['medhinc'] = (X_train['medhinc'] - medmean) / medstd
    X_train['walkscore'] = (X_train['walkscore'] - walmean) / walstd
    X_train['elem_score'] = (X_train['elem_score'] - elemean) / elestd
    X_test['medhinc'] = (X_test['medhinc'] - medmean) / medstd
    X_test['walkscore'] = (X_test['walkscore'] - walmean) / walstd
    X_test['elem_score'] = (X_test['elem_score'] - elemean) / elestd    

    k = 14 #initially we'll look at the six (6) closes neighbors

    keys = find_nearest_neighbors_Q7(X_train, X_test, k)

    predictions_Q7 = []
    for key in keys:
        prediction = sum(key.values * y_train.iloc[key.index.values]) # SUM OF (weights (AKA key.values) * y_train)
        predictions_Q7.append(prediction)


    n_question7 = len(y_test)
    vars = 3 #the number of independent variables in the model
    SSR_question7 = sum((predictions_Q7-y_test)**2)
    TSS_question7 = sum((y_test-np.mean(y_test))**2)

    r_squared_question7 = 1-(SSR_question7)/(TSS_question7)
    print('Q7 - The r_squared is {}'.format(round(r_squared_question7, 3)))
          
    
Question_7()
    

