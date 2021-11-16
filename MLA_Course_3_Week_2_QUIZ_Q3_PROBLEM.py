#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor # fifth question
from sklearn.pipeline import make_pipeline # fifth question
from sklearn.preprocessing import StandardScaler # fifth question
from sklearn.model_selection import cross_val_score # fifth question

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



def question_3():
	
	def gradientDescent(X,y,lossFunction):
		
		learning_rate = 0.01 
		
		t = 0 # Then we initialize the iteration counter to zero
		budget = 5000 # We set a max number of iterations to stop at
		min_iterations = 50 # We also set a min number of iterations to try
		
		w = np.zeros(X.shape[1])# We initialize w to a matrix of 0s with the shape of X
		
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
		print(loss)
		gradient = 2*(X.T.dot(error))/m # The gradient is the derivative of the loss function
	#	print(len(gradient))
		return (loss, gradient)
	
	
	def assymSquaredLoss(X,y,prediction):
		m = len(y)
		alpha = 0.05
		error = prediction-y
		loss = np.mean(np.where(prediction <= y, alpha * (prediction - y)**2, (prediction - y)**2 ))
		gradient = 2*(X.T.dot(error))/m # The gradient is the derivative of the loss 
		
		return(loss, gradient)
	
	w_gradient_descent, losses = gradientDescent(X_train, y_train, assymSquaredLoss)
	print('Q3: ', w_gradient_descent.elem_score, sep='\n')   #22JAN: answer of 2.776 (WRONG)  
	
question_3()