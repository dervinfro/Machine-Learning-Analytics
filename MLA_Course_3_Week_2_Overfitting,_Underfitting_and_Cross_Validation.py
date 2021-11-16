import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures # second section
from sklearn.model_selection import cross_val_score # second section

from sklearn.pipeline import make_pipeline # third section

from sklearn.model_selection import GridSearchCV # fourth section

from sklearn.model_selection import KFold # fifth section

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/sampledata.csv')
df['log_value'] = np.log(df['value'])
X = df['rooms']
y = df['log_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=10) #setting the random state so that each time this code is ran, we pull the same samples in the same order.


def firstSection():
    
    m = 1 # m is the polynomial to start from
    max_poly = 30 #max_poly is the max degree of polynomial to try
    
    # These lists keep track of the Mean Squared Error (MSE) for each polynomial
    training_scores = [0]
    test_scores = [0]
    
    # These arrays will hold the independent variables we'll use.  Initially, it will just be a column of ones (1's).  
    # Each time we loop through, we'll add a colun X_train**m and X_test**m
    X_train_polys = np.ones(shape=(len(X_train),1))
    X_test_polys = np.ones(shape=(len(X_test),1))
    
    # Each loop adds a column to our arrays X_train**m and X_test**m
    # It fits a simple linear regression, calculates the R squared for the train and test sets
    # Those R squared values are then added to the lists above
    while m != (max_poly+1):
        X_train_polys = np.append(X_train_polys, (X_train**m).values.reshape(-1,1), axis=1)
        X_test_polys = np.append(X_test_polys, (X_test**m).values.reshape(-1,1), axis=1)
        reg = LinearRegression().fit(X_train_polys, y_train)
        
        training_score = reg.score(X_train_polys, y_train)
        test_score = reg.score(X_test_polys, y_test)
        training_scores.append(training_score)
        test_scores.append(test_score)
        m = m+1
     
     
    
    plt.xlabel('m')
    plt.ylabel('r squared')
    plt.plot(training_scores, label='training')
    plt.plot(test_scores, label='test')
    plt.legend()
    plt.show()
    
#firstSection()

##################
##################
    
def secondSection():

    print('Second Section')
    
    m = 3 # We will use the 3rd degree polynomial
    folds = 5 # We will use 5 folds for cross-validation
    
    polys = PolynomialFeatures(m)
    X_train_polys = polys.fit_transform(X_train.values.reshape(-1,1))
    reg = LinearRegression()
    
    val_scores = cross_val_score(reg, X_train_polys, y_train, cv=5)
    
    print('The mean values for r square from cross validation are: ')
    print(np.mean(val_scores))
    print('Their standard deviation is {}'.format(np.std(val_scores)))
    
    # If these results are satisfactory, we can fit our model on the entire training set
    reg = LinearRegression().fit(X_train_polys, y_train)
    train_score = reg.score(X_train_polys, y_train)
    print('The r squared for the full training set is {}'.format(round(train_score, 6)))
    
    # When we go to test score, we need to apply the same transformation without refitting
    X_test_polys = polys.transform(X_test.values.reshape(-1,1))
    test_score = reg.score(X_test_polys, y_test)
    print('The r squared for the test set is {}'.format(round(test_score,6)))
    print('')
    
#secondSection()

##################
##################

def thirdSection():
    
    print('Third Section')
    
    m = 3 # We will use the 3rd generation polynomial
    folds = 5 # We will use 5 folds for cross-validation
    
    # Create a pipeline that first adds teh m polynomil features and then fits a linear regression using them
    pipeline = make_pipeline(PolynomialFeatures(m), LinearRegression())
    
    #During cross-validation, in each fold, the pipeline is applied to the test set and then tested on the validation set
    val_score = cross_val_score(pipeline, X_train.values.reshape(-1,1), y_train, cv=folds)
    print(val_score.mean())
    print('')
    
#thirdSection()

##################
##################

def fourthSection():
    
    print('Fourth Section')
    
    folds = 5 # Set the numbe of folds for the GridSearchCV to use
    
    # Set the degrees for GridSearchCV to try -- as above, we'll try 1-30
    degrees = np.arange(30)
    p_grid = {'polynomialfeatures__degree': degrees}
    
    # Create pipeline, which first adds polynomials to the model and then fits it to linear regression
    pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())
    
    # Create the GridSearchCV with our pipeline, the grid of parameters to try, and the folds
    grid_search = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=folds)
    grid_search.fit(X_train.values.reshape(-1,1), y_train)
    
    test_score = grid_search.score(X_test.values.reshape(-1,1), y_test)
    
    print('The best parameters of the grid search are: ', grid_search.best_params_)
    print('The score of the best parameter of grid search is: ', grid_search.best_score_)
    print('The score of the best parameter of the grid search on the test set is: ', test_score)
    
#fourthSection()

##################
##################

def fifthSection():
     
    
    folds = 5
    trials = 40
    
    degrees = np.arange(30)
    p_grid = {'polynomialfeatures__degree': degrees}
    
    pipeline = make_pipeline(PolynomialFeatures(), LinearRegression())
    
    nested_scores = []
    unnested_scores = []
    
    for i in range(trials):
        outer_cv = KFold(n_splits=folds, shuffle=True)
        inner_cv = KFold(n_splits=folds, shuffle=True)
        
        grid_search = GridSearchCV(estimator=pipeline, param_grid=p_grid, cv=inner_cv)
        grid_search.fit(X_train.values.reshape(-1,1), y_train)
        unnested_score = grid_search.best_score_
        unnested_scores.append(unnested_score)
        nested_score = cross_val_score(grid_search, X_train.values.reshape(-1,1), y_train, cv=outer_cv).mean()
        nested_scores.append(nested_score)
        
    print('fifth section')
    print('The mean nested cv score is: ', np.mean(nested_scores))
    print('The mean unnested score is: ', np.mean(unnested_scores))
    
fifthSection()
      
    
    