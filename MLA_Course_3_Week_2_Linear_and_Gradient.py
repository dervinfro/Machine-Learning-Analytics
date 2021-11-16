import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def gradientDescent(X, y, lossFunction):
    learning_rate = .01 #first....we set the learning rate
    
    t = 0 # then....we initialize the iteration counter to 0
    budget = 2000 # we set the maximum number of iterations to stop at
    min_iterations = 50 # we also set a minimum number of iterations to try
    
    w = np.zeros(X.shape[1]) # we initialize w to a matrix of 0s with the shape of X
    
    mindelta = 1.001  # if the change in loss between the current iteration and the previous one falls below this fraction, we stop
    
    convered = False # the change does fall below min_delta, convered is set to True and the loop stops
    
    losses = [] # Finally, this list is created to keep track of the loss in each iteration
    
    while (t!=budget) and (convered!=True):
        
        prediction = X.dot(w)
        loss, gradient = lossFunction(X, y, prediction)
        losses.append(loss)
        
        w = w-learning_rate * gradient # w is adjusted based on the learning rate and gradient
        
        t = t+1
        
        if (t>min_iterations) and (losses[t - min_iterations]/loss<mindelta):
            convered=True
            
            
    return (w, losses)

#For this exercise, we'll use the mean squared error as the loss function

def squaredLoss(X, y, prediction):
    m = len(y)
    error = prediction -y
    loss = np.mean(error**2)
    gradient = 2*(X.T.dot(error))/m #the gradient is the derivative of the loss function
    return (loss, gradient)

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/sampledata.csv')

df['log_value'] = np.log(df['value'])
df['const'] = 1 # the first column in the array needs to be populated with 1's to account for the intercept

# for this sample, we'll run a bivariate regression with the independent variable the number of fireplaces.
X = df[['const','fireplaces']]
y = df['log_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25) # setting the random state for reproducability

w_gradient_descent, losses = gradientDescent(X_train, y_train, squaredLoss)

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.show()


###############
###############

w_closed_form = np.dot(np.linalg.pinv(X_train),y_train)

print('using the closed form solution, the estimate for w are: ')
for w in w_closed_form:
    print(w)
    
print('using gradient descent, the estimates for w are: ')
for w in w_gradient_descent:
    print(w)
    
    