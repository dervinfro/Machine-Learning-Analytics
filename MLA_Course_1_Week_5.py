import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time
import scipy.stats as stats
import statsmodels.formula.api as sm
import pingouin as pg


start_time = time.time()
df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/batterycountvstemp.csv')
#sns.scatterplot(y = df['Battery_Count'], x = df['Temperature'])


def long_hand_ols():
	#LONG HAND MATH FOR OLS REGRESSION
	covariance_xy = 0
	list_covar_xy = []

	#Temperature = independent variable (X-Axis)
	#Battery_Count = dependent variable (Y-Axis)

	for x, y in zip(df['Temperature'], df['Battery_Count']):
		covariance_xy = (x-df['Temperature'].mean())*(y-df['Battery_Count'].mean()) + covariance_xy
	#	list_covar_xy.append((x-df['Temperature'].mean())*(y-df['Battery_Count'].mean()))
	print(covariance_xy) # time: 0.1396331787109375
	#print(sum(list_covar_xy)) #time: 0.14014387130737305

	variance_x = 0
	list_variance_x = []

	for x in df['Temperature']:
		variance_x = (x - df['Temperature'].mean())**2 + variance_x
		list_variance_x.append((x - df['Temperature'].mean())**2)

	long_hand_ols.beta_one = covariance_xy/variance_x
	long_hand_ols.beta_zero = (df['Battery_Count'].mean() - long_hand_ols.beta_one * df['Temperature'].mean())
	print('The OLS Regresssion is y(hat) = {} + {} * x '.format(long_hand_ols.beta_zero, long_hand_ols.beta_one))

long_hand_ols()


def ols_regression():
	#PYTHON LIBRARY FOR OLS REGRESSION (SCIPY.STATS.LINREGRESS)
	b_one, b_zero, r_val, p_val_beta_1, stderr_beta_1 = stats.linregress(x = df['Temperature'], y = df['Battery_Count'])

	print('The OLS Regression (scipy stats library) is y(hat) = {} + {} * x '.format(b_zero, b_one))

	#graph the regression
	real_xs = df['Temperature'] #real x values
	ols_regression.predicted_ys = [] # predicted y values (list)

	for x in real_xs:
		predicted_y = long_hand_ols.beta_zero + long_hand_ols.beta_one * x
		ols_regression.predicted_ys.append(predicted_y)

	#sns.lineplot(x=df['Temperature'],y=predicted_ys, color='red', label='Regression Line')
	#sns.scatterplot(y = df['Battery_Count'], x = df['Temperature'])
	#plt.show()
ols_regression()


def bivarte_regressions():
	#MEASURE AND TEST THE BIVARITE REGRESSIONS (SECOND LAB OF WEEK 5)
	ESS = 0 #explained sum of squares
	TSS = 0 #total sum of squares

	#ESS
	for x in df['Temperature']:
		y_hat = long_hand_ols.beta_zero + long_hand_ols.beta_one * x
		ESS = (y_hat - df['Battery_Count'].mean()) ** 2 + ESS
	print('ESS: {}'.format(ESS))

	#TSS
	for y in df['Battery_Count']:
		TSS = (y - df['Battery_Count'].mean()) ** 2 + TSS
		
	#R Squared = gives a measurement on how well the regression explains the dependent (Battery Count (Y-Axis)) variable.
	#the closer R2 is to the value of 1; the better the regression is at making predictions for the Y-Axis (Battery_Count).
	# the .34 means that this regression explains 34% of the variance in the dependent variable.
	R_Squared = ESS/TSS #R Squared: 0.3408089912760396
	print('R Squared: {}'.format(R_Squared))
	
bivarte_regressions()


def measure_precision():
	#MEASURING PRECISION
	#standard error of regression (SER)
	#sum of square residuals (SSR)
	n = len(df)
	SSR = 0


	for x,y in zip(df['Temperature'], df['Battery_Count']):	
		ser_y_hat = long_hand_ols.beta_zero + long_hand_ols.beta_one * x
		SSR = (y - ser_y_hat)**2 + SSR
	SER = (SSR/n-2)**(1/2)
	print('The SER is: {} '.format(SER)) #The SER is: 23.221645244099676
	#this relates to the standard deviation.  
	#68% +/- 1 standard deviation.  In this case 68% of the observations are within 23 counts above/below the regression
	#95% +/- 2 standard deviations. In this case 95% of the observations are within 46 counts above/below the regression.
	#99% +/- 3 " 																	"69"								"

	y_2std_above = ols_regression.predicted_ys + 2 * SER
	y_2std_below = ols_regression.predicted_ys - 2 * SER
	sns.lineplot(x=df['Temperature'], y=ols_regression.predicted_ys, color='blue', label='Regression Line')
	sns.lineplot(x=df['Temperature'], y = y_2std_above, color='red', label='2 STD Above Reg. Line')
	sns.lineplot(x=df['Temperature'], y = y_2std_below, color='red', label='2 STD Above Reg. Line')
	sns.scatterplot(x=df['Temperature'], y=df['Battery_Count'])
	plt.title('2 Standard Deviations')
	#plt.show()
	
measure_precision()

#statsmodels using Python Library
model = sm.ols(formula='Battery_Count ~ Temperature', data=df) #give the names of the columns and the dataframe associated with it
results = model.fit()
print(results.summary())
n = len(df)
SER = (results.ssr/n-2)**(1/2)
print('SER: ', SER)


print("*" * 25,'\n',"*" * 25,'\n')

pg_model = pg.ttest(df['Battery_Count'], df['Temperature'] )
print(pg_model)


end_time = time.time()
print('time: {}'.format(end_time - start_time))