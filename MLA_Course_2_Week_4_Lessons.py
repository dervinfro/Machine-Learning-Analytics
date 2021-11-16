import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

'''
Missingness, Outliers, and Precision
'''

df=pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/project2data.csv')
df = df.dropna()
model = sm.ols('np.log(value) ~ avg_school_score + walkscore + np.log(medhinc)', data=df)
result = model.fit()
print(result.summary())

print('\n')
result_huber_white = model.fit(cov_type='HC0') #in reference to the Huber-White (a different way to calculate standard error in the presence of heteroskedasticity)
print(result_huber_white.summary())

print('\n')
result_clustered = model.fit(cov_type='cluster', cov_kwds={'groups':df['tri']}) #in reference to clustered data points (geographically clustered data is divided into three categories)
print(result_clustered.summary())