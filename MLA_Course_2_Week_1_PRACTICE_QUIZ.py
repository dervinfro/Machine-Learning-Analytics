import pandas as pd
import numpy as np
from scipy.stats import f

'''
Practice Quiz - Course 2 Week 1
'''

def Question_5():
	'''
	Question 5:
	R^2=.74
	The sample size is 587
	number of independent variables in the model is 4. 
	What is the adjusted R^2 (Report to three decimal places).
	'''
	n = 587
	k = 4
	r_squared = .74
	adj_rsquared = 1 - ((1 - r_squared)*(n - 1)/(n - k -1))
	print('Q 5: ', round(adj_rsquared, 3))
	print('R Squared (Adjusted) - Numerator: ', (n-1)/(n-k-1)) #this equation did not work for me.  I pulled the above equation from: https://www.statisticshowto.com/adjusted-r2/
Question_5()

def Question_5_Example():
	'''
	NOTE: see the second lab notebook (MEASURING AND TESTING MULTIVARIATE OLS) ; res4 model for data for this example.  R Squared = 0.463659 // Adj. R Squared = 0.452050
	0.452050 = X * 0.536341
	'''
	n = 237
	k = 5
	r_squared = 0.463659
	
	adj_rsquared = 1 - ((1 - r_squared)*(n - 1))/(n - k - 1)
	print('Q 5 (Example Data): ', adj_rsquared)
	
Question_5_Example()


def Question_6_7_8():
	'''
	Question 6:
	'''
	SSR_unrestrict = 31021 #return value for the unrestricted model, using all of the variables. (years tenure; years education; performance rating; number of projects)
	SSR_restrict = 32129 #return value for the restricted model, using performance rating and number of projects
	total_samples = 187
	k = 4 #See variables in unrestricted model
	q = 2 #count the number of equals signs OR the number of variables in the joint hypothesis 
	f_stat = ((SSR_restrict - SSR_unrestrict)/q)/(SSR_unrestrict/(total_samples - k - 1))
	print("Q 6: ", f_stat)
	
	p_val = f.sf(3.0455878083194268, q, total_samples-k-1)
	print('Q 7: ', p_val)
	
	target_p_val = f.ppf(.95, q, total_samples-k-1)
	print('Q 8: ', target_p_val)
Question_6_7_8()

#Question 8: Hand jammed floats in place of variable f_stat.  3.04 returns a p_val of 0.05
# my initial response above (3.04) was incorrect.  The quiz wanted the full value of 3.0455878083194268

def Question_10():
	R_Squared = .8
	VIF = 1/(1 - R_Squared) # question - what kind of R_Squared value would result in a VIF(Variable Influence Factor) score of 5
	print('Q 10: ', VIF)
Question_10()	
