import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
import statsmodels.api as sm_api
from sklearn.linear_model import LinearRegression

pd.options.display.float_format = '{:.2f}'.format


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/project2data.csv')

def multivariate_regressions():
	#print(df.info())
#	print(df['value'].describe())
	df['Log_Value'] = np.log(df['value'])
	model = sm.ols(formula='Log_Value ~ high_school_score + elem_score', data=df)
	result = model.fit()
	print(result.summary())
	print('*' * 25)


multivariate_regressions()

def statsmodels_api():
	
	X = df[['high_school_score', 'elem_score']]
	X = sm_api.add_constant(X) #NOTE: that with the statsmodels.api, the constant term needs to be added manually
	y = df['Log_Value']
	model1 = sm_api.OLS(y,X) #statsmodels.api OLS does not use string to specify regression
	result1 = model1.fit()
	print(result1.summary())
	print('*' * 25)
	
statsmodels_api()

def sklearn_linear_regression():
	X = df[['high_school_score', 'elem_score']]
	X = sm_api.add_constant(X) #NOTE: that with the statsmodels.api, the constant term needs to be added manually
	y = df['Log_Value']
	model1 = LinearRegression()
	model1.fit(X,y)
	betas = model1.coef_
	intercept = model1.intercept_
	r = model1.score(X,y)
	print("R-Squared: ", r) #RESULT: 0.20476638927289426 (this score is the same value as R-Squared in the above model.summary()
	
sklearn_linear_regression()

print(df['tri'].unique())

dummies = pd.get_dummies(df['tri'])
print(dummies.sample(5))



#########################
## START OF 2ND MODULE###
#########################


	