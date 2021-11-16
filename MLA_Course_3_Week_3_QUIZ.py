import pandas as pd
import numpy as np
import timeit

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge # question 2
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score # question 1
from sklearn.metrics import mean_squared_error # question 1
from sklearn.preprocessing import MinMaxScaler # question 2
from sklearn.pipeline import make_pipeline # question 2

pd.options.display.max_rows=None
pd.options.display.max_columns=None
pd.options.display.width= 175

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/quiz3part1data.csv')
df['const'] = 1
df['log_value'] = np.log(df['value'])
y = df['log_value']
X = df[['const', 'medhinc', 'fireplaces', 'garagesize', 'walkscore', 'numreviews_med',
'dist_med', 'dist_med_lessthan1k', 'elem_score', 'high_school_score',
'beds', 'college', 'lon', 'lat', 'white', 'black', 'totalpop',
'repair', 'garageconstruction', 'garageconstruction2']]

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.25, random_state=25)

# Q1 Linear Regression
def question_1():
    '''
    Then, create a OLS model, and score it using 2-fold cross-validation. You may have noticed that cross_val_score() has different scoring options. This time, report the mean of the negative mean squared errors (sklearn uses negative error so that a higher score is better). Round to four decimal places.
    '''
    folds = 2
    ols_regression = LinearRegression()
    val_scores = cross_val_score(ols_regression, X_train, y_train, cv=folds, scoring='neg_mean_squared_error')
    mean_val_scores = np.mean(val_scores)
    print('Q1: ', round(mean_val_scores,4)) #correct output is: -0.1307
    print('')
    
question_1()

# Q2 Ridge
def question_2and3():
    '''
    QUESTION 2:
    Now, create a Ridge regression. Use box normalization to scale the features first so that they fall between 0 and 1 (use MinMaxScaler() in sklearn.preprocessing). Then, use Grid Search with 2-fold cross-validation to find the best value for alpha, again scoring with negative mean squared error. Search for alphas in np.logspace(-5, 5, 100).

    What is the best value for alpha that you find? Round to three decimal places.
    
    QUESTION 3:
    What's the mean negative mean squared error of the best model that Grid Search finds for the Ridge regression? Round to four decimal places.

    '''
    #NOTE: See the following link for example on GridSearchCV and Ridge: https://stackoverflow.com/questions/57376860/how-to-run-gridsearchcv-with-ridge-regression-in-sklearn
    
    folds = 2
    alphas = np.logspace(-5,5,100)
  
    parameters ={'ridge__alpha':alphas} # NOTE: ensure that this KEY in this dictionary is listed as VALID KEY under:     print(ridge_pipeline.get_params().keys())
    ridge_pipeline = make_pipeline(MinMaxScaler(feature_range=(0,1)), Ridge(fit_intercept=True)) #Scaled data (ie MinMaxScaler) CANNOT be passed into GridSearchCV as a parameter.  PIPELINE MUST BE USED.
    print(ridge_pipeline.get_params().keys())
    
    ridge_grid_search = GridSearchCV(estimator=ridge_pipeline, param_grid=parameters, scoring='neg_mean_squared_error', cv=folds)
    
    ridge_grid_search.fit(X_train, y_train)
    print('Q2: ', ridge_grid_search.best_params_) #NOT: -0.135 // -0.131 // -0.136  ***  Correct: Q2:  Ridge(alpha=1.1233240329780265)
    print('Q3: ', round(ridge_grid_search.best_score_,4)) # Correct: -0.1306 
    
    
question_2and3()



# Q4 & Q5 - Lasso
def question_4and5and6():
    '''
    
    '''
    folds = 2
    alphas = np.logspace(-5,5,100)
    
   
    
    
    parameters ={'lasso__alpha':alphas} # NOTE: ensure that this KEY in this dictionary is listed as VALID KEY under:     print(lasso_pipeline.get_params().keys())
    lasso_pipeline = make_pipeline(MinMaxScaler(feature_range=(0,1)), Lasso())
    
    lasso_grid_search = GridSearchCV(estimator=lasso_pipeline, param_grid=parameters, scoring='neg_mean_squared_error', cv=folds)
    lasso_best_model = lasso_grid_search.fit(X_train, y_train)
    print('Q4: ', lasso_best_model.best_params_) # Correct: {'lasso__alpha': 0.00025950242113997375}
    print('Q5: ', round(lasso_best_model.best_score_,4)) # Correct: -0.1306
    print('Q6: ', X.columns, lasso_best_model.best_estimator_.named_steps.lasso.coef_) 
    
question_4and5and6()
'''
Q1:  -0.1307
Q2:  {'ridge__alpha': 1.1233240329780265}
Q3:  -0.1306
Q4:  {'lasso__alpha': 0.00025950242113997375}
Q5:  -0.1306
Q6:  Index(['const', 'medhinc', 'fireplaces', 'garagesize', 'walkscore', 'numreviews_med', 'dist_med', 'dist_med_lessthan1k', 'elem_score', 'high_school_score', 'beds', 'college',
       'lon', 'lat', 'white', 'black', 'totalpop', 'repair', 'garageconstruction', 'garageconstruction2'],
      dtype='object') [ 0.          0.04538632  1.27313984  0.16952655  0.17021245  1.17637655
 -0.0079063  -0.00931533  0.08229199  0.07494255  1.62414662  1.29130337
 -0.21990826  0.5169903   0.12317919 -0.13647087 -0.08505689 -0.26766749
  0.02441777  0.        ]
'''





