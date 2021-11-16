import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

pd.options.display.max_columns = None

df = pd.read_csv ('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/quiz2.csv')

df['value'] = np.log(df['value'])
df['squarefoot'] = np.log(df['squarefoot'])
#df.insert(20,'squarefoot_log', np.log(df['squarefoot']))


############
#QUESTION 1#
############
def Question_One():
	'''
	What is the Adjusted R^2?
	'''
	
	model = sm.ols(formula='value ~ squarefoot + homeowner + avg_school_score + homeowner * avg_school_score + walkscore + I(walkscore**2) + I(walkscore**3)', data=df)
	result = model.fit()
#	print('Q1 (Summary):, ', result.summary()) # this is just to verify the r squared below.
	print('Q1: ', round(result.rsquared_adj, 3)) #0.564 (CORRECT)
	print('*' * 23)
	print('')
Question_One()


############
#QUESTION 2#
############
def Question_Two():
	global df

	model = sm.ols(formula='value ~ squarefoot + homeowner + avg_school_score + homeowner * avg_school_score + walkscore + I(walkscore**2) + I(walkscore**3)', data=df)
	result_q2= model.fit()
	print('Q2:', result_q2.params, sep='\n')
	
	#params commented below and then transposed to hard variables.
	#Intercept                     0.170179
	#squarefoot                    0.594451
	#homeowner                     0.000835
	#avg_school_score              1.191252
	#homeowner:avg_school_score    0.005843
	#walkscore                    -0.015301
	#I(walkscore ** 2)            -0.000059
	#I(walkscore ** 3)             0.000003

	b_zero = 0.170179
	b_one = 0.594451
	b_two = 0.000835
	b_three = 1.191252
	b_four = 0.005843
	b_five = -0.015301
	b_six = -0.000059
	b_seven = 0.000003

	'''
       below is in specific reference to section: https://znjvtwht.labs.coursera.org/notebooks/Module_2/Transformations.ipynb#Interpreting-Polynomial-Models
    this is how I interpret this problem is solved....however, this is not the correct answer.
    '''

	log_value = b_zero + b_one + b_two * (1) + b_three + b_four * (1) + b_five + b_six + b_seven - (b_zero + b_one + b_two * (0) + b_three + b_four * (0) - b_five - b_six + b_seven)

	#the two following variables are a break down of the above single variable.
	log_value1 = b_zero + b_one + b_two * (1) + b_three + b_four * (1) - b_five - b_six + b_seven
	diff = b_zero + b_one + b_two * (0) + b_three + b_four * (0) - b_five - b_six + b_seven

	print('log value: ', log_value)
	print('log value(percentage): ', '{:.2%}'.format(log_value)) #percentage of log_value
	print('log value and diff: ', log_value1, diff) # output of the two seperate variables
	print('log value minus diff: ', log_value1 - diff) #this output is equal to log_value.  Used for my confirmation purposes.
	print('log value/diff -1: ', (log_value1/diff) - 1) # equation format as specified in the details of the Q2 problem.
	q2_ouput = (log_value1/diff) -1
	print('log value/diff -1 (percentage): ', '{:.2%}'.format(q2_ouput))
	print('')
	
	'''
	NOTE: Below is the correct code for Question 2 (9JAN2021) - See Tim's notebook that he passed me with his correct answers.
	
	use the model from above and set two samples: ( 
	homeowner = 0 WHERE avg school value = 7
	homeowner = 1 WHERE avg school value = 7
	'''
	
	out_of_samples_value1 = {'avg_school_score':7, 'homeowner':1, 'squarefoot':0, 'walkscore':0, 'I(walkscore ** 2)':0, 'I(walkscore ** 3)':0}
	v1 = result_q2.predict(out_of_samples_value1)
	
	out_of_samples_value0 = {'avg_school_score':7, 'homeowner':0, 'squarefoot':0, 'walkscore':0, 'I(walkscore ** 2)':0, 'I(walkscore ** 3)':0}
	v0 = result_q2.predict(out_of_samples_value0)
	
	print('Q2 (v1):',v1.to_string(), sep='\n')
	print('Q2 (v0):',v0.to_string(), sep='\n')
	
	print('')
Question_Two()



############
#QUESTION 3#
############
def Question_Three():
	unrestricted_model = sm.ols(formula='value ~ squarefoot + homeowner + avg_school_score + homeowner * avg_school_score + walkscore + I(walkscore**2) + I(walkscore**3)', data=df)
	unrestricted_result = unrestricted_model.fit()
	ssr_ur = sum(unrestricted_result.resid**2)
	
	restricted_model = sm.ols(formula='value ~ squarefoot + homeowner + avg_school_score + homeowner * avg_school_score', data=df)
	restricted_result = restricted_model.fit()
	ssr_r = sum(restricted_result.resid**2)
	
	k = 7 #number of variables in the unrestricted model
	q = 3 #number of equal signs in the restriction **OR** the number of Betas (coefficients)
	n = len(df) # total number of samples
	
	f_stat = ((ssr_r - ssr_ur)/q)/(ssr_ur/(n-k-1))
	print('Q3: ', round(f_stat,2)) #473.22 (CORRECT)
Question_Three()

############
#QUESTION 4#
############
def Question_Four():
	'''
	A 5% (0.05) increase in squarefootage is equal to X change in value
	squarefoot field is a position 19 (insert new field at 20)
	'''
	model_q4 = sm.ols('value ~ squarefoot + homeowner + avg_school_score + homeowner * avg_school_score + walkscore + I(walkscore**2) + I(walkscore**3)', data=df)
	Question_Four.result_q4 = model_q4.fit()
	print(Question_Four.result_q4.params)
	print('*' * 23)
	print('Q4: (squarefoot) ', Question_Four.result_q4.params['squarefoot'])
	print('')
	
	
	q4_sqf_adj = Question_Four.result_q4.params['squarefoot']
	print('Q4 (as a percentage): ',round(q4_sqf_adj * 5, 2))
	print('')
	#0.00 (Incorrect) diff between intercept coefficients
	#5.00 (Incorrect) diff between Beta1 of top model and Beta1 of the model in question.
	#2.28 (Correct)	As per the Q4 in the practice quiz, this will be 5 * Beta1 (5 * 0.4563236138754279 = 2.28)
Question_Four()





############
#QUESTION 5#
############
def Question_Five():
	global df
	
	model_q4 = Question_Four.result_q4
	
	k = 7 #number of variables in the unrestricted model
	q = 3 #number of equal signs in the restriction **OR** the number of Betas (coefficients)
	n = len(df) # total number of samples
	
	SER = (model_q4.ssr/(n-k-1))**(1/2)
	print('Q5: ', round(SER, 2)) #0.52
Question_Five()