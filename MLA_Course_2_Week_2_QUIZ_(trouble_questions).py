#!/usr/bin/env python3

import pandas as pd
import numpy as np
import statsmodels.formula.api as sm
from scipy import stats

pd.options.display.max_columns = None

df = pd.read_csv ('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/quiz2.csv')
df['value'] = np.log(df['value'])
df['squarefoot'] = np.log(df['squarefoot'])
#NOTE: These models MUST BE RUN with the log conversions above.


############
#QUESTION 2#
############
def Question_Two():
	
    global df
    #df = df[df['avg_school_score'] == 7]

    model = sm.ols(formula='value ~ squarefoot + homeowner + avg_school_score + homeowner * avg_school_score + walkscore + I(walkscore**2) + I(walkscore**3)', data=df)
    result_q2= model.fit()
    print('Q2: \n', result_q2.params)
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
 
    log_value = b_zero + b_one + b_two * (1) + b_three + b_four * (1) - b_five - b_six + b_seven - (b_zero + b_one + b_two * (0) + b_three + b_four * (0) - b_five - b_six + b_seven)

    #the two following variables are a break down of the above single variable.
    log_value1 = b_zero + b_one + b_two * (1) + b_three + b_four * (1) - b_five - b_six + b_seven
    diff = b_zero + b_one + b_two * (0) + b_three + b_four * (0) - b_five - b_six + b_seven

    print('log value: ', log_value)
    print('log value(percentage): ', '{:.2%}'.format(log_value)) #percentage of log_value
    print('log value and diff: ', log_value1, diff) # output of the two seperate variables
    print('log value minus diff: ', log_value1 - diff) #this output is equal to log_value.  Used for my confirmation purposes.
    print('log value/diff -1: ', (log_value1/diff) - 1) # equation format as specified in the details of the Q@ problem.
    q2_ouput = (log_value1/diff) -1
    print('log value/diff -1 (percentage): ', '{:.2%}'.format(q2_ouput))	


Question_Two()


############
#QUESTION 4#
############
'''
def Question_Four():
	
	#A 5% (0.05) increase in squarefootage is equal to X change in value
	#squarefoot field is a position 19 (insert new field at 20)
	
	
	print(df.shape)
	print(df['avg_school_score'].value_counts)
	
	model_q4 = sm.ols('value ~ squarefoot + homeowner + avg_school_score + homeowner * avg_school_score + walkscore + I(walkscore**2) + I(walkscore**3)', data=df)
	result_q4 = model_q4.fit()
	print('Q4: ', result_q4.params)
	print('*' * 23)
	
	df.insert(21,'squarefoot_times_5percent', df['squarefoot']*1.05)
	model_q4_sqf_adjusted = sm.ols('value ~ squarefoot_times_5percent + homeowner + avg_school_score + homeowner * avg_school_score + walkscore + I(walkscore**2) + I(walkscore**3)', data=df)
	result_q4_sqf_adjusted = model_q4_sqf_adjusted.fit()
	print('*' * 23)
	print('Q4 (SQF Adjusted):', result_q4_sqf_adjusted.params)
	
	q4_sqf = 0.456324
	q4_asqf_adj = 0.434594
	print('Q4 (as a percentage): ', round(((q4_sqf/q4_asqf_adj) - 1) * 100, 3))
	
	k = 7 #number of variables in the unrestricted model
	q = 3 #number of equal signs in the restriction **OR** the number of Betas (coefficients)
	n = len(df) # total number of samples
	
	SER = (result_q4_sqf_adjusted.ssr/(n-k-1))**(1/2)
	print('Q5: ', round(SER, 2)) #0.52
	
Question_Four()
'''
