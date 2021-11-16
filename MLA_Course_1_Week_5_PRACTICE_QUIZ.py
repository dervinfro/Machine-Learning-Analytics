import pandas as pd
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.formula.api as sm
import pingouin as pg


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/interestanddiff.csv')

#X AXIS = Interest (independent variable)   
#Y AXIS = Diff (dependent variable)

#Q4 - What value to do you calculate for Beta One?
#**SHORT HAND VERSION**
beta_one, beta_zero, r_val, p_val_beta_1, stederr_beta_1 = stats.linregress(x=df['Interest'], y=df['Diff'])
print(beta_one, beta_zero, r_val, p_val_beta_1, stederr_beta_1)
print('Beta One: {} Beta Zero: {} P val beta 1: {} '.format(beta_one, beta_zero, p_val_beta_1)) 
#VALUES: Beta One: 0.09041857118433558 Beta Zero: 2.455884839497733 P val beta 1: 0.00024000650411751931

#**LONG HAND VERSION**
list_covar_xy = [] #covariance xy list

for x, y in zip(df['Interest'], df['Diff']):
	list_covar_xy.append((x-df['Interest'].mean())*(y-df['Diff'].mean()))
	
covar_xy = sum(list_covar_xy) 



list_variance_x = [] #variance x list

for x in df['Interest']:
	list_variance_x.append((x-df['Interest'].mean())**2)

var_x = sum(list_variance_x) #VALUE: 3534.09390410959



beta_1 = covar_xy/var_x
print('Beta 1: ', beta_1) 

beta_0 = df['Diff'].mean() - beta_1 * df['Diff'].mean()
print('Beta 0: ', beta_0)


real_xs = df['Diff'] #real x values
predicted_ys = [] #predicted y values (list)

for x in real_xs:
	predicted_y = beta_0 + beta_1 * x #predicted y value (looped)
	predicted_ys.append(predicted_y)
	
print('OLS Regression is y(hat): {} + {} * x'.format(beta_0, beta_1))

sns.lineplot(x=df['Diff'],y=predicted_ys, color='Red', label='Regression Line') #predicted line plot
sns.scatterplot(x=df['Diff'], y=df['Interest']) 
#plt.show()

#MEASURE AND TEST THE BIVARIATE OF REGRESSIONS
ESS = 0 #Explained Sum of Squares
TSS = 0 #Total Sum of Squares

#ESS (Explained Sum of Squares)
for x in df['Interest']:
	y_hat = beta_zero + beta_one * x
	ESS = (y_hat - df['Diff'].mean()) ** 2 + ESS
print('ESS: ', ESS) 

#TSS (Total Sum of Squares)
for y in df['Diff']:
	TSS = (y - df['Diff'].mean()) ** 2 + TSS
print('TSS: ', TSS) 


#Q5 - What is the R^2 of this regression?
R_Squared = ESS/TSS
print('Q5 - R Squared: ', R_Squared)



#Q6 - What is the p_value for Beta One?
#SEE OLS Regression Results - VALUE: 0.0


#Q7 - What is the Standard Error of Regression?
#SER: Standard Error of Regression (How far does the observed values deviate from the line (SEE: predicted_ys in sns.lineplot)
SSR = 0 #Sum of Square Residuals
n = len(df)

for x,y in zip(df['Interest'], df['Diff']):
	ser_y_hat = beta_zero + beta_one * x
	SSR = (y - ser_y_hat)**2 + SSR
SER = (SSR/(n-2)) ** (1/2)
print('SER: ', SER)


model = sm.ols(formula='Interest ~ Diff', data=df)
results = model.fit()
print(results.summary())





	 