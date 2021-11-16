import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC # Question 4
from sklearn.multiclass import OneVsRestClassifier # Question 4
from sklearn.pipeline import make_pipeline # Question 4
from sklearn.svm import SVC # question 5
from sklearn.model_selection import GridSearchCV # Question 5
from sklearn.neighbors import KNeighborsClassifier # Question 8

'''
Sandals are label 5.

Sneakers are label 7.

Ankle boots are label 9
'''

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/fashion_MNIST_shoes.csv')
print(df.head(10))

dummies = pd.get_dummies(df['label'])

X = df.drop("label", axis=1) # default is axis=0 (AKA: Rows).  Axis=1 tell us df['label'] is a column.  If axis=1 is not included it will kick out an error.
y = dummies
print(X.head(10))
print(y.head(10))


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)


def question_1():
    '''
    Use the exact same SGD algorithm we wrote to create a multi-class classifier, with the same starting parameters: max_epochs = 15, batch_size=32, and lr=.01.

    What accuracy do you calculate for your training data?
    '''
    def softmax(scores):
        scaled_scores = scores-scores.max(axis=1, keepdims=True) #subtract the maximum to prevent overflow
        exp_scores = np.exp(scaled_scores)
        softmax_scores = exp_scores/(np.sum(exp_scores,axis=1, keepdims=True))
        return softmax_scores
    
    def get_log_loss(X,w,y):
        scores = np.dot(X,w)
        predictions = softmax(scores)
        log_likelihood = np.sum(y*np.log(predictions))/len(y)
        log_loss = -log_likelihood
        return log_loss
    
    def get_gradient(X,w,y):
        scores = np.dot(X,w)
        predictions = softmax(scores)
        error = y - predictions
        
        gradient = -(np.dot(X.T, error))/len(y)
        return gradient
    
    def SGD_softmax(X, y, lr, batch_size, max_epochs):
        w = np.zeros([X.shape[1], y.shape[1]])
        old_loss = 1000
        losses = []
        target = .01
        
        count = 0
        while (count < max_epochs):
            shuffled_index = np.random.permutation(X.shape[0])
            batch_starts = range(0, X.shape[0], batch_size)
            
            for start_index in batch_starts:
                batch = shuffled_index[start_index:start_index + batch_size]
                x_batch = X[batch]
                y_batch = y[batch]
                gradient = get_gradient(x_batch, w, y_batch)
                w = w - lr * gradient
                
            current_loss = get_log_loss(X, w, y)
            gain = (old_loss - current_loss)/np.abs(old_loss)
            losses.append(current_loss)
            old_loss = current_loss
            
            if (gain<target):
                lr = lr/2
                
                
            count = count+1
            
        return (w, losses)
    
    max_epochs = 15
    batch_size = 32
    lr=.01
    X_scaled = StandardScaler().fit_transform(X_train)
    
    const = np.ones((X_scaled.shape[0],1))
    X_biased = np.concatenate([const,X_scaled],1)
    
    w, losses = SGD_softmax(X_biased, y_train.values, lr, batch_size, max_epochs)  
    
    predictions = np.argmax(np.dot(X_biased, w), axis=1)
    actual = np.argmax(y_train.values, axis=1)
    right = 0
    i = 0
    while (i<len(y_train)):
        if predictions[i]==actual[i]:
            right = right + 1
        i = i+1
    print("Q1 - Accuracy: ", right/len(y_train))
    #Q1 = 0.97 **CORRECT**
    

question_1()

def question_2():
    '''
    Now, make predictions on the test set. Remember to scale the data appropriately.

    What's the accuracy for the test set? Again, round to two decimal places.
    '''
    
    def softmax(scores):
        scaled_scores = scores-scores.max(axis=1, keepdims=True) #subtract the maximum to prevent overflow
        exp_scores = np.exp(scaled_scores)
        softmax_scores = exp_scores/(np.sum(exp_scores,axis=1, keepdims=True))
        return softmax_scores
    
    def get_log_loss(X,w,y):
        scores = np.dot(X,w)
        predictions = softmax(scores)
        log_likelihood = np.sum(y*np.log(predictions))/len(y)
        log_loss = -log_likelihood
        return log_loss
    
    def get_gradient(X,w,y):
        scores = np.dot(X,w)
        predictions = softmax(scores)
        error = y - predictions
        
        gradient = -(np.dot(X.T, error))/len(y)
        return gradient
    
    def SGD_softmax(X, y, lr, batch_size, max_epochs):
        w = np.zeros([X.shape[1], y.shape[1]])
        old_loss = 1000
        losses = []
        target = .01
        
        count = 0
        while (count < max_epochs):
            shuffled_index = np.random.permutation(X.shape[0])
            batch_starts = range(0, X.shape[0], batch_size)
            
            for start_index in batch_starts:
                batch = shuffled_index[start_index:start_index + batch_size]
                x_batch = X[batch]
                y_batch = y[batch]
                gradient = get_gradient(x_batch, w, y_batch)
                w = w - lr * gradient
                
            current_loss = get_log_loss(X, w, y)
            gain = (old_loss - current_loss)/np.abs(old_loss)
            losses.append(current_loss)
            old_loss = current_loss
            
            if (gain<target):
                lr = lr/2
                
                
            count = count+1
            
        return (w, losses)
    
    max_epochs = 15
    batch_size = 32
    lr=.01
    X_scaled = StandardScaler().fit_transform(X_test)
    
    const = np.ones((X_scaled.shape[0],1))
    X_biased = np.concatenate([const,X_scaled],1)
    
    w, losses = SGD_softmax(X_biased, y_test.values, lr, batch_size, max_epochs)  
    
    predictions = np.argmax(np.dot(X_biased, w), axis=1)
    actual = np.argmax(y_test.values, axis=1)
    right = 0
    i = 0
    while (i<len(y_test)):
        if predictions[i]==actual[i]:
            right = right + 1
        i = i+1
    print("Q2 - Accuracy: ", right/len(y_test)) 
    # answer is 0.9126984126984127....but this is as per Tim.  I'm not getting this.  Not sure WTF is going on.
    
question_2()

def question_3():
    '''
    Incorporate regularization, and consider the following values for lambda: [0, .001, 1.0, 10.0].

    Which results in the best accuracy score on the test set? (note that you would not follow this procedure in practice: you would tune your parameters on a validation set, and only score on the test set when you're finished configuring your model.)
    '''
    def softmax(scores):
        scaled_scores = scores-scores.max(axis=1, keepdims=True) #subtract the maximum to prevent overflow
        exp_scores = np.exp(scaled_scores)
        softmax_scores = exp_scores/(np.sum(exp_scores,axis=1, keepdims=True))
        return softmax_scores
    
    def get_log_loss(X,w,y):
        scores = np.dot(X,w)
        predictions = softmax(scores)
        log_likelihood = np.sum(y*np.log(predictions))/len(y)
        log_loss = -log_likelihood
        return log_loss
    
    def get_gradient(X,w,y):
        scores = np.dot(X,w)
        predictions = softmax(scores)
        error = y - predictions
        
        regularizer = np.vstack((np.zeros((1,w.shape[1])),w[1:,:])) # this line of code is specific to Q3
        
        gradient = -(np.dot(X.T, error))/len(y) + lamb*regularizer # this line of code is specific to Q3  
        
        #gradient = -(np.dot(X.T, error))/len(y) #this code was the original code used in Q2
        return gradient
    
    def SGD_softmax(X, y, lr, batch_size, max_epochs, lamb): #"lamb" was added to this function for Q3.
        w = np.zeros([X.shape[1], y.shape[1]])
        old_loss = 1000
        losses = []
        target = .01
        
        count = 0
        while (count < max_epochs):
            shuffled_index = np.random.permutation(X.shape[0])
            batch_starts = range(0, X.shape[0], batch_size)
            
            for start_index in batch_starts:
                batch = shuffled_index[start_index:start_index + batch_size]
                x_batch = X[batch]
                y_batch = y[batch]
                gradient = get_gradient(x_batch, w, y_batch)
                w = w - lr * gradient
                
            current_loss = get_log_loss(X, w, y)
            gain = (old_loss - current_loss)/np.abs(old_loss)
            losses.append(current_loss)
            old_loss = current_loss
            
            if (gain<target):
                lr = lr/2
                
                
            count = count+1
            
        return (w, losses)
    
    lr=.01
    batch_size = 32
    max_epochs = 15
    lamb = .001 # test values for lamb are: [0, .001, 1.0, 10.0]
    
    
    X_scaled = StandardScaler().fit_transform(X_test)
    
    const = np.ones((X_scaled.shape[0],1))
    X_biased = np.concatenate([const,X_scaled],1)
    
    w, losses = SGD_softmax(X_biased, y_test.values, lr, batch_size, max_epochs, lamb) #"lamb" was added to this function for Q3.
    
    predictions = np.argmax(np.dot(X_biased, w), axis=1)
    actual = np.argmax(y_test.values, axis=1)
    right = 0
    i = 0
    while (i<len(y_test)):
        if predictions[i]==actual[i]:
            right = right + 1
        i = i+1
    print("Q3 - Accuracy: ", right/len(y_test), 'Lamb: ', lamb)  
    # Q3 - Accuracy:  0.9576719576719577 Lamb:  0
    # Q3 - Accuracy:  0.9603174603174603 Lamb:  0.001 **CORRECT**
    # Q3 - Accuracy:  0.9312169312169312 Lamb:  1.0
    # Q3 - Accuracy:  0.8703703703703703 Lamb:  10.0
    
    
question_3()

def question_4():
    '''
    Train a one-vs-all SVM linear classifier. Use scikit-learn's helpful sklearn.multiclass.OneVsRestClassifier class and its sklearn.svm.LinearSVC class. Use the parameters random_state=1 and C=1 for the linear svm, and leave the other parameters on their default settings. Note that LinearSVC fits an intercept by default for you, so don't worry about including a constant in your matrix.

    Again, make sure to use StandardScaler fitted to the training set. What is the accuracy on the test set? Round to two decimal places.
    '''
    
    model = make_pipeline(StandardScaler(), OneVsRestClassifier(LinearSVC(C=1, random_state=1, fit_intercept=True)))
    model.fit(X_train, y_train)
    print('Q4: ', model.score(X_test,y_test)) # The .score attribute returns the mean accuracy
    # Q4:  0.8333333333333334 ** CORRECT **
    
question_4()


def question_5():
    '''
    Question 5
    
    You now want to try some kernels. You're not sure about whether you should use a polynomial or rbf, and you're not sure what parameters you should use for them. So, you want to use GridSearchCV to find the best kernel and the best parameters.
    
    Again, use a one-vs-all classifier, and for the SVM use use sklearn.svm.SVC, with random_state=1. (note by the way that you can use this class for multiclass classification by itself as well, but it uses a one-vs-one strategy).
    
    Use GridSearchCV with 2-fold validation to find the best estimator.
    
    For the parameter grid, test kernels rbf and poly.
    
    For the rbf kernel:
    
    For the values of C, test 1, 10, and 100
    
    For the values for gamma, test .001, .01, and .1
    
    For the poly kernel:
    
    For the values of C, test 1, 10, and 100
    
    For the values for degree, test 2 and 3
    
    Again, use StandardScaler(). As you'll recall, you'll want to use a pipeline. 
    
    Two hints here. (1) For your grid search, you may need to use my_grid_search.estimator.get_params().keys() to figure out what to name the parameters in the parameter grid. (2) Take a look at this example if you get stuck.
    '''
    
    parameters_both = {'onevsrestclassifier__estimator__kernel':['poly','rbf'], 
                       'onevsrestclassifier__estimator__C':(1, 10, 100), 
                       'onevsrestclassifier__estimator__gamma':(.001, .01,.1),
                       'onevsrestclassifier__estimator__degree': [2,3]}    
    
    svc_pipeline = make_pipeline(StandardScaler(), OneVsRestClassifier(SVC(random_state=1)))
    #print(svc_pipeline.get_params().keys())
    grid_search = GridSearchCV(estimator=svc_pipeline, param_grid=parameters_both, cv=2, refit=True)
    grid_search.fit(X_train, y_train)
    print('Q5: ', grid_search.score(X_test, y_test)) # Q5:  0.8862433862433863 **CORRECT Answer is: C for a value of 100
    print('Q7: ', grid_search.best_score_, grid_search.best_estimator_) # Q7:  0.8992932862190812 ** CORRECT **
    print(grid_search.best_params_)
    
question_5()

def question_8():
    '''
    Question 8

    One last item to investigate. We actually have known a multiclass classifier since the first week of class: k-nearest neighbors. We used it for regression then, but the process for classification is very similar.
    
    Instead of using a system to average values, a system is used to count the neighbors' labels as votes. If, for example, you're doing 5-nearest neighbors, your prediction might be the most commonly occurring label among the 5 nearest neighors.
    
    So, you want to now use GridSearch to experiment with this classification system. Use sklearn.neighbors.KNeighborsClassifier. Aside for the values for k, use default parameters (meaning that it will use Euclidean distance and uniform weights).
    
    For k, search for values 1-15. Again, use StandardScaler and 2-fold cross-validation. What's the best score on the training set? Round to two decimal places.
    '''

    pipeline = make_pipeline(StandardScaler(), KNeighborsClassifier(weights='uniform', p=2))
    
    degrees = np.arange(1,16)
    parameters = {'kneighborsclassifier__n_neighbors': degrees}
     
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters, cv=2)
    grid_search.fit(X_train,y_train)
    print('Q8: ', grid_search.best_score_) # Q8:  0.8692579505300353 **CORRECT** as of 5FEB
    #print('Q8 (Estimator for K): ', grid_search.best_estimator_)
    #print('Q8 cv results: ', grid_search.cv_results_)

question_8()
'''
Q1 - Accuracy:  0.9708480565371025
Q2 - Accuracy:  0.9497354497354498
Q3 - Accuracy:  0.9497354497354498 Lamb:  0.001
Q4:  0.8333333333333334
Q5:  0.8862433862433863
Q7:  0.8992932862190812
Q8:  0.8692579505300353
'''




