import pandas as pd
import numpy as np
import statsmodels.formula.api as sm

#pd.options.display.max_columns=None
pd.options.display.max_rows=None

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/Wrightsample.csv')

'''
REF for graphic on diff in diff (DiD): https://towardsdatascience.com/causal-inference-101-difference-in-differences-1fbbb0f55e85
'''


#print(df.loc[:, ['carthefts_pc_post','carthefts_pc_post_mean']])
#for x in range(len(df.columns)):
    #print(x, df.columns[x])

'''
Q1: What's the difference between the mean of the post-barrier carthefts per capita in the outside localities and the mean of the pre-barrier carthefts per capita in the outside localities? Round to three decimal places.
'''
outer = df[df['TREATB'] == 'Outside']
north = df[df['TREATB'] == 'North']
south = df[df['TREATB'] == 'South']

      
q1_result = outer['carthefts_pc_post'].mean() - outer['carthefts_pc_pre'].mean() # columns 187 & 188
print('Q1: ', round(q1_result, 3)) 
#CORRECT: 0.006

'''
Q2: Using just conditional means--the mean of the car thefts for the north pre- and post-barrier, and the mean for the outside localities pre- and post-barrier--calculate the DiD using the outside localities as the control. Round to three decimal places?
'''

north_pre = north[north['PT'] == 'Pre-barrier']
north_post = north[north['PT'] == 'Post-barrier']

outer_pre = outer[outer['PT'] == 'Pre-barrier']
outer_post = outer[outer['PT'] == 'Post-barrier']

Q2_c_minus_a = north_post['carthefts_pc'].mean() - north_pre['carthefts_pc'].mean() #treatment group is A -> C
Q2_d_minus_b = outer_post['carthefts_pc'].mean() - outer_pre['carthefts_pc'].mean() #control group is B -> D
Q2_diff_in_diff = Q2_c_minus_a - Q2_d_minus_b #DiD = treatment group - control group
print('Q2: ', round(Q2_diff_in_diff, 3))
#INCORRECT: -0.053
#CORRECT: -0.516


'''
Q3: Using just conditional means, what is the DiD comparing the outside localities to the southern localities in mean carthefts per capita pre- and post-barrier, using the outside localities as a control? Round to three decimal places.
'''

south_pre = south[south['PT'] == 'Pre-barrier']
south_post = south[south['PT'] == 'Post-barrier']

Q3_c_minus_a = south_post['carthefts_pc'].mean() - south_pre['carthefts_pc'].mean() #treatment group is A -> C
Q3_d_minus_b = outer_post['carthefts_pc'].mean() - outer_pre['carthefts_pc'].mean() #control group is B -> D
Q3_diff_in_diff = Q3_c_minus_a - Q3_d_minus_b #DiD = treatment group - control group
print('Q3: ', round(Q3_diff_in_diff, 3))
#CORRECT: 0.339

'''
Q4: Create a regression to detremine the DiD comparing the north (the treatment group) to the outside (the control group). Add to the model the same additional variables that Austin and his co-authors introduce in their study. It should take the form:

carthefts_pc = beta_0 + beta_1Treat + beta_2Post + beta_3Treat*Post + popN + regional_council + urban + distance2greenline + distance2greenline2 + Time

Note that the variable for the comparison of north to the outside--in the model specified above, the variable named "Treat"--is TN in the dataset.

What's the coefficient for \beta_3β 
​	
 ? Round to three decimal places.
'''


q4_category_var = df.loc[:, ['carthefts_pc','TN','PT','popN','regional_council','urban','distance2greenline','distance2greenline2','Time']] #this var was used to confirm all variable were indeed quantative (AKA numerical) values and NOT categorical variables
#print(q4_category_var)
#print(df.loc[:, ['TN','TREATB']]) #used to compare the difference between the two variables.
#tn_north = df[df['TN'] == 1.0]
#tn_outside = df[df['TN'] == 0.0]
#df['tn_north'] = tn_north['TN']
#df['tn_outside'] = tn_outside['TN']

dummies = pd.get_dummies(df['PT'])
df['PostBarrier'] = dummies['Post-barrier']


model_q4 = sm.ols('carthefts_pc ~ TN + PostBarrier + TN * PostBarrier + popN + regional_council + urban + distance2greenline + distance2greenline2 + Time', data=df)
result_q4 = model_q4.fit()
print('Q4: ', round(result_q4.params[3], 3))
#CORRECT: -0.519

'''
Q5: You want to look at the standard error to detremine statistical significance. What's the clustered standard error for Beta_3 in the model that you created above? Round to three decimal places.

Note that some of the variables in the model may have NaN values that need to be dropped to calculate the clustered standard error.
'''

df_dropped_q5 = df.dropna(subset=['TN','PostBarrier','popN','regional_council','urban','distance2greenline','distance2greenline2','Time']) 

model_q5 = sm.ols('carthefts_pc ~ TN + PostBarrier + TN * PostBarrier + popN + regional_council + urban + distance2greenline + distance2greenline2 + Time', data=df_dropped_q5)
result_q5 = model_q5.fit(cov_type='cluster', cov_kwds={'groups': df_dropped_q5['lcode']})
print('Q5 (Standard Error): ', round(result_q5.bse[3],3)) #bse prints out 'std err' column
#INCORRECT: 0.003
#CORRECT: 0.102

'''
Q6: So, you've compared the north to the south and the north to the outside localities. In both cases, the north's crime, as measured by carthefts per capita, seems to have decreased post-barrier. But a question remains: could trends in the south have changed post-barrier?

Conduct a final DiD experiment where you compare the south to the outisde localities. It should have the same specification as before:

carthefts_pc = beta_0 + beta_1Treat + beta_2Post + beta_3Treat*Post + popN + regional_council + urban + distance2greenline + distance2greenline2 + Time


Note that the variable for the comparison of south to the outside--in the model specified above, the variable named "Treat"--is TS in the dataset.

What's the coefficient for \beta_3β 
3
​	
 ? Round to three decimal places.
'''
model_q6 = sm.ols('carthefts_pc ~ TS + PostBarrier + TS * PostBarrier + popN + regional_council + urban + distance2greenline + distance2greenline2 + Time', data=df)
result_q6 = model_q6.fit()
print('Q6: ', round(result_q6.params[3],3))
#CORRECT: 0.339

'''
Q7: What is the clustered standard error for Beta_3in the model specified above? Round to three decimal places.
'''
df_dropped_q7 = df.dropna(subset=['TS','PostBarrier','popN','regional_council','urban','distance2greenline','distance2greenline2','Time']) 

model_q7 = sm.ols('carthefts_pc ~ TS + PostBarrier + TS * PostBarrier + popN + regional_council + urban + distance2greenline + distance2greenline2 + Time', data=df_dropped_q7)
result_q7 = model_q7.fit(cov_type='cluster', cov_kwds={'groups': df_dropped_q7['lcode']})
#print('Q7: ', result_q7.summary())
print('Q7 (Standard Error): ', round(result_q7.bse[3], 3)) #bse prints out 'std err' column
#INCORRECT: 0.003
#CORRECT: 0.109
