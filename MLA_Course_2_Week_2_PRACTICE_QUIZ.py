import pandas as pd
import numpy as np

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/project1data.csv')

'''
Question 7: (ALL OTHER QUESTIONS WERE MULTI-CHOICE.  SEE SCREEN SHOTS FOR CORRECT ANSWERS.)
You have created the following model:

\hat{log\_value} = 12.4384 + 0.1483*avg\_school\_score - 0.0471*walk\_score + 0.0005*walk\_score^2 

 

All things being equal, what is the marginal effect on the estimate of log_value if you increase walkscore by 2 units, starting from a value of 80 (i.e., the effect of moving from a walkscore of 80 to 82)? Report to three decimal places.
'''

beta_zero = 12.4384
beta_one = 0.1483
beta_two = 0.0471
beta_three = 0.0005
avg_ss = 10 # See notes below for explation on this variable.
walk_s = (80 + 2)

#beta_zero + beta_one * avg_ss - beta_two * 80  + beta_three * 80**2

log_value = beta_zero  + beta_one * avg_ss - beta_two * (80 + 2) + beta_three * (80 + 2)**2 
diff = beta_zero + beta_one * avg_ss - beta_two * 80  + beta_three * 80**2
print('Log: ', log_value)
print('Diff: ', diff)
print(round(log_value - diff, 3))

'''
In ref to the notes for variable "avg_ss", this one initally confused me.  However, after a few iterations I'd realized that regardless of the interger that I had entered for this variable, my outcome was always the same answer.  That lead me to believe, that I could enter any value here and it would not effect my answer in a negative way.
'''