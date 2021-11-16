import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor # fifth question
from sklearn.pipeline import make_pipeline # fifth question
from sklearn.preprocessing import StandardScaler # fifth question
from sklearn.model_selection import cross_val_score, GridSearchCV # fifth question

pd.options.display.max_columns=None
pd.options.display.max_rows=None
pd.options.display.width=175
#pd.options.display.float_format='{:.3f}'.format

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/quiz2data.csv')

# As per the instructions, the dependent variable is log of value
df['log_value'] = np.log(df['value'])

#we need a constant variable in OLS models (SEE: https://www.theanalysisfactor.com/the-impact-of-removing-the-constant-from-a-regression-model-the-categorical-case/)
df['const'] = 1

X = df[['const','elem_score','homeowner']]
y = df['log_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)



##################
## QUESTION ONE ##
##################
def question_1():

    def gradientDescent(X,y,lossFunction):
        
        learning_rate = 0.01 
        
        t = 0 # Then we initialize the iteration counter to zero
        budget = 5000 # We set a max number of iterations to stop at
        min_iterations = 50 # We also set a min number of iterations to try
        
        w = np.zeros(X.shape[1])# We initialize w to a matrix fo 0s with the shape of X  (NOTE: shape[1] is the number of columns // shape[0] is the number of rows
        
        mindelta = 1.00001 # If the change between the current iteration and the previous one falls below this value, we stop
        
        converged = False
        
        losses = []
        
        while (t!=budget) and (converged!=True): # The loop continues until we hit the iteraion budget, or converged is True
            prediction = X.dot(w)
            loss, gradient = lossFunction(X,y,prediction)
            losses.append(loss)
            
            w = w - learning_rate*gradient # w is adjusted based on the learning rate and gradient
            t = t+1
            
            if (t>min_iterations) and (losses[t - min_iterations]/loss<mindelta):
                converged = True
                
        return (w, losses)
    
    def squaredLoss(X,y,prediction):
        m = len(y)
        error = prediction - y
        loss = np.mean(error**2) # Loss is the mean square error
        gradient = 2*(X.T.dot(error))/m # The gradient is the derivative of the loss function
        return (loss, gradient)
        
    w_gradient_descent, losses = gradientDescent(X_train, y_train, squaredLoss)
    print('Q1:', round(w_gradient_descent.elem_score,3))
    
    ##################
    ## QUESTION TWO ##
    #################

    #What does the closed-form solution give as the value for the coefficient for elem_score? Round to three decimal places.

    w_closed_form = np.dot(np.linalg.pinv(X_train), y_train)
    
    print('Q2: ', round(w_closed_form[1], 3)) # The [1] value is the elem_score value of the list - w_closed_form
        
question_1()
    

    
    
####################
## QUESTION THREE ##
####################

#Now, instead of using the squared loss function, use the asymmetric squared loss function. Set a value for alpha of .05. 
#What value do you now estimate for the coefficient for elem_score? Round to three decimal places.

def question_3():
    
    def gradientDescent(X,y,lossFunction):
        
        learning_rate = 0.01 
        
        t = 0 # Then we initialize the iteration counter to zero
        budget = 5000 # We set a max number of iterations to stop at
        min_iterations = 50 # We also set a min number of iterations to try
        
        w = np.zeros(X.shape[1])# We initialize w to a matrix fo 0s with the shape of X  (NOTE: shape[1] is the number of columns // shape[0] is the number of rows
        
        mindelta = 1.00001 # If the change between the current iteration and the previous one falls below this value, we stop
        
        converged = False
        
        losses = []
        
        while (t!=budget) and (converged!=True): # The loop continues until we hit the iteraion budget, or converged is True
            prediction = X.dot(w)
            loss, gradient = lossFunction(X,y,prediction)
            losses.append(loss)
            
            w = w - learning_rate*gradient # w is adjusted based on the learning rate and gradient
            t = t+1
            
            if (t>min_iterations) and (losses[t - min_iterations]/loss<mindelta):
                converged = True
                
        return (w, losses)    
    
    def assymSquaredLoss(X,y,prediction):
        m = len(y)
        alpha = 0.05
        error = prediction-y
        l = np.where(error <=0, alpha * ((prediction - y)**2), (prediction - y)**2)
        loss = np.mean(l)
        weight = np.where(error <= 0, alpha, 1)
        gradient = 2*(X.T.dot(weight * error))/m # The gradient is the derivative of the loss
        return(loss, gradient)
    
    w_gradient_descent, losses = gradientDescent(X_train, y_train, assymSquaredLoss)
    print('Q3: ', w_gradient_descent.elem_score)
    
    
question_3()
    

###################
## QUESTION FOUR ##
###################



def question_4():
    
    def gradientDescent(X,y,lossFunction):
        
        learning_rate = 0.058 #lowest learning rate to get the model to converge before it hits the budget limit of 5000 iterations. 
        
        t = 0 # Then we initialize the iteration counter to zero
        budget = 5000 # We set a max number of iterations to stop at
        min_iterations = 50 # We also set a min number of iterations to try
        
        w = np.zeros(X.shape[1])# We initialize w to a matrix fo 0s with the shape of X  (NOTE: shape[1] is the number of columns // shape[0] is the number of rows
        
        mindelta = 1.00001 # If the change between the current iteration and the previous one falls below this value, we stop
        
        converged = False
        
        losses = []
        
        while (t!=budget) and (converged!=True): # The loop continues until we hit the iteraion budget, or converged is True
            prediction = X.dot(w)
            loss, gradient = lossFunction(X,y,prediction)
            losses.append(loss)
            
            w = w - learning_rate*gradient # w is adjusted based on the learning rate and gradient
            t = t+1
            
            if (t>min_iterations) and (losses[t - min_iterations]/loss<mindelta):
                converged = True
                
        print('Q4: T: {} // Converged: {} // Learning Rate: {} '.format(t, converged, learning_rate)) # This line outputs the 't' iteration number as well as the boolean value of Budget
                
        return (w, losses)    
    
    def assymSquaredLoss(X,y,prediction):
        m = len(y)
        alpha = 0.05
        error = prediction-y
        l = np.where(error <=0, alpha * ((prediction - y)**2), (prediction - y)**2)
        loss = np.mean(l)
        weight = np.where(error <= 0, alpha, 1)
        gradient = 2*(X.T.dot(weight * error))/m # The gradient is the derivative of the loss
        return(loss, gradient)
    
    w_gradient_descent, losses = gradientDescent(X_train, y_train, assymSquaredLoss)
    
    
question_4()

###################
## QUESTION FIVE ##
###################

def question_5():
    '''
    This time around, we'll use cross validation to tune our model before scoring it on the test set.

We'll use the dependent variable log_value, and the independent variables medhinc, walkscore, and elem_score.

First, divide your data into a train and test set, with a 75/25 split, with random_state=25.

This time, use Scikit-learn's KNeighborsRegressor. 

You'll want to scale your data, but remember that you want to use the validation set the same way as you would use your test set. The parameters used for scaling should be independent of the values that are held out. So, you'll want to use a pipeline with StandardScaler and cross-validation to tune your model.

First, as a test, create a pipeline with a k-NN model using weighted Manhattan distance and k=7 and StandardScaler. Conduct a simple cross-validation using 5 folds, wiith no shuffling (i.e., just use cross_val_score with cv=5). What's the mean R-squared you get for cross-validation? Round to three decimal places.
    '''
    
    X = df[['const','medhinc','elem_score','walkscore']]
    y = df['log_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)    
    pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=7, weights='distance', p=1))

    val_score = cross_val_score(pipeline, X_train, y_train, cv=5)
    print('Q5 Val Score Mean: ', round(val_score.mean(),3))



question_5()

###################
## QUESTION SIX ##
###################
def question_6thru8():
    '''
    Use Grid Search with cv=5 to find the best value for k. What is it? You can access the best estimator in your grid search with grid_search.best_estimator_
    '''
    
    X = df[['const','medhinc','elem_score','walkscore']]
    y = df['log_value']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)
    
    pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(weights='distance', p=1))
    #print(sorted(pipeline.get_params().keys()))  #this params key IS PARAMOUNT to understand....it MUST correspond to the 'parameters' dictionary key.  SEE: kneighborsregressor__n_neighbors
    # https://stackoverflow.com/questions/34889110/random-forest-with-gridsearchcv-error-on-param-grid
    
    degrees = np.arange(1,31)
    parameters = {'kneighborsregressor__n_neighbors': degrees}
    
    grid_search = GridSearchCV(pipeline, parameters, cv=5)
    grid_search.fit(X_train, y_train)
    print('Q6: ', grid_search.best_estimator_)   # answer is 10 
    
    test_score = grid_search.score(X_test, y_test)
    print('Q7: ', round(grid_search.best_score_,3))
    print('Q8: ', round(test_score,3))    

question_6thru8()
'''
ANSWERS:
Q1: -0.066
Q2:  -0.067
Q3:  0.5998244471222773
Q4: T: 4990 // Converged: True // Learning Rate: 0.058 
Q5 Val Score Mean:  0.47
Q6:  Pipeline(steps=[('standardscaler', StandardScaler()),
                ('kneighborsregressor',
                 KNeighborsRegressor(n_neighbors=10, p=1, weights='distance'))])
Q7:  0.482
Q8:  0.534
'''



