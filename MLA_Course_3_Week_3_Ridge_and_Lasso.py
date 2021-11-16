import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

pd.options.display.max_rows=None
pd.options.display.max_columns=None
pd.options.display.width= 175

alphas = np.logspace(-10,10,100)
df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/project3data.csv')
df = df.dropna()

variables = ["medhinc", "squarefoot", "homeowner","garagearea", "walkscore", "rooms",
             "poverty", "elem_score", "high_school_score"]

X = df[variables]
y = np.log(df["value"])

#As Greg mentioned, you need to scale your features.
X_scaled = StandardScaler().fit_transform(X)

def ridge_reg(): # ridge regression
    ridge_coef = []
    
    for alpha in alphas:
        ridge = Ridge(alpha, fit_intercept=True)
        ridge.fit(X_scaled,y)
        ridge_coef.append(ridge.coef_)
        
    ax = plt.gca()
    ax.set_xscale('log')
    ax.plot(alphas, ridge_coef)
    ax.legend(variables)
    plt.tight_layout()
    #plt.show()
    
    print('ridge:', ridge_coef[99])
ridge_reg()

def lasso_reg():    
    
    lasso_coefs = []
    
    for alpha in alphas:
        lasso = Lasso(alpha, fit_intercept=True)
        lasso.fit(X_scaled,y)
        lasso_coefs.append(lasso.coef_)
        
    ax = plt.gca()
    ax.set_xscale('log')
    ax.plot(alphas, lasso_coefs)
    ax.legend(variables)
    plt.tight_layout()
    #plt.show()
    
lasso_reg()
    
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=10)

p_grid={'lasso__alpha':alphas}
lasso_pipeline = make_pipeline(StandardScaler(), Lasso(fit_intercept=True))
grid_search = GridSearchCV(estimator=lasso_pipeline, param_grid=p_grid, cv=5)
grid_search.fit(X_train,y_train)
print('The best paramter for the Lasso regression is:', grid_search.best_params_)
print('The best score on the training set is: ', grid_search.best_score_)
print('The score on the test set using the best parameter is: ', grid_search.score(X_test,y_test))

best_estimator_coef = grid_search.best_estimator_.named_steps.lasso.coef_
print('For the best estimator, the coefficients are: ', best_estimator_coef, sep='\n')
    
    

