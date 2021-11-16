import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.misc import derivative	
from scipy.stats import norm
from scipy.stats import logistic



pd.options.display.max_columns = None 

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/weidmannandcallen.csv')
df = df.dropna()
'''
(327, 18) Index(['Unnamed: 0', 'distid', 'distname', 'regcommand', 'fraudchi2_total',
       'est_fraud_share', 'fraudchi2_k', 'fraudchi2_a', 'v.elday.pc',
       'v.elday.pc.2', 'v.2months.pc', 'v.2months.pc.2', 'v.elday.pc.lag',
       'numclosedstations', 'electrified', 'pce_1000_07', 'distkabul',
       'elevation'],
      dtype='object')
'''
##############
# Question 1 #
##############

'''
What's the coefficient for violence in the 2 month period leading up the election for your LPM? Round to three decimal places.
'''

Y = df['fraudchi2_total'] #dependent variable
X = df[['v.2months.pc','v.2months.pc.2', 'numclosedstations','electrified','pce_1000_07','elevation']] #a total of six independent variables.  v.2months.pc = Violence in two months preceding the election
X = sm.add_constant(X) # Reason for needing a constant: https://www.theanalysisfactor.com/the-impact-of-removing-the-constant-from-a-regression-model-the-categorical-case/
LPM_Model = sm.OLS(Y, X)
LPM_Result = LPM_Model.fit()
print('Q1 Params: ', LPM_Result.params[1]) 


##############
# Question 2 #
##############
    
'''
According to this model, what is the marginal effect of an increase in violence in v.2months.pc around (+- 0.5) its mean? Round to three decimal places
REF: Numerical Approx. Marginal Effects (https://www.aptech.com/blog/marginal-effects-of-linear-models-with-data-transformations/)

'''

q2_mean = df['v.2months.pc'].mean()


q2_final_answer = LPM_Result.params[1] + 2 * LPM_Result.params[2] * q2_mean #0.259 (See the Chegg email with the tutor and the link above.)

print('Q2 Marginal Effects: ', round(q2_final_answer, 3)) 
'''
0.595 - Wrong
0.297 - Wrong
0.077 - Wrong
0.179 - Wrong
'''
##############
# Question 3 #
##############
'''
Assume that if your model gives a .5 probability or greater, then it predicts that fraud occurred; if it predicts a probability less than .5, then it predicts that fraud did not occur.

What is the ratio of correct predictions to the total number of observations used for the LPM model?
'''

LPM_predicted_probablities = LPM_Result.predict()
LPM_predicted_outcomes = []
for x in LPM_predicted_probablities:
    if x >= .5:
        LPM_predicted_outcomes.append(1) # 1 means that fraud occured
    else:
        LPM_predicted_outcomes.append(0) # 0 means that fraud did not occur

#for x in range(len(Y)):
    #print(df['fraudchi2_total'].iloc[x], LPM_predicted_probablities[x], LPM_predicted_outcomes[x]) # this a comparisson of fields for my sanity
    
LPM_hits = 0
i = 0

for fraud in Y:
    if LPM_predicted_outcomes[i] == fraud:
        LPM_hits = LPM_hits + 1
    i = i + 1
        
LPM_percent_correct = LPM_hits/len(Y)
print('Q3: ', round(LPM_percent_correct, 3)) #0.856

##############
# Question 4 #
##############
'''
Create a probit model using the same variables. What coefficient does it estimate for v.2months.pc?
'''
    
Probit_Model = sm.Probit(Y, X)
Probit_Result = Probit_Model.fit()
print('Q4 Params: ', round(Probit_Result.params[1],3)) #v.2months.pc         1.212
print('Q4 Summary: ', Probit_Result.summary())

##############
# Question 5 #
##############
'''
What is the partial effect at the average (PEA) of v.2months.pc?
'''

linear_model = Probit_Result.params[0] + Probit_Result.params[1]*df['v.2months.pc'].mean() + Probit_Result.params[2]*df['v.2months.pc.2'].mean() + Probit_Result.params[3]*df['numclosedstations'].mean() + Probit_Result.params[4]*df['electrified'].mean() + Probit_Result.params[5]*df['pce_1000_07'].mean() + Probit_Result.params[6]*df['elevation'].mean()
PEA = norm.pdf(linear_model, 0, 1)*Probit_Result.params[1]
print('Q5: ', round(PEA, 3)) 
'''
0.34 - Wrong ( Probit_Result.params[4]*df['electrified'] was missing out of the linear model )
0.231
'''

##############
# Question 6 #
##############
'''
Assume that if your model gives a .5 probability or greater, then it predicts that fraud occurred; if it predicts a probability less than .5, then it predicts that fraud did not occur.
'''
Probit_predicted_poss = Probit_Result.predict()
Probit_predicted_outcomes = []
for x in Probit_predicted_poss:
    if x >= 0.5:
        Probit_predicted_outcomes.append(1)
    else:
        Probit_predicted_outcomes.append(0)
        
Probit_hits = 0
probit_i = 0

for fraud_q6 in Y:
    if Probit_predicted_outcomes[probit_i] == fraud_q6:
        Probit_hits = Probit_hits + 1
    
    probit_i = probit_i + 1
    
Probit_precent_correct = Probit_hits/len(Y)
print('Q6 ', round(Probit_precent_correct, 3)) 
'''
0.853 (Wrong - for the Probit_i counter, I was mistakenly using the 'i' from the LPM model)
0.859
'''

##############
# Question 7 #
##############
'''
Create a classifier that uses logit. What is the average partial effect (APE) for v.2months.pc?
'''
Logit_Model = sm.Logit(Y,X)
Logit_Result = Logit_Model.fit()

linear_model_Logit = X.dot(Logit_Result.params)
pdfs_Logit = logistic.pdf(linear_model_Logit)
partial_effects_Logit = pdfs_Logit*Logit_Result.params[1]
APE = partial_effects_Logit.mean()
print('Q7: ', round(APE, 3)) 
'''
0.238 (wrong - I used norm.pdf when it should be logistic.pdf)
0.209 (wrong - I had Probit_Result instead of Logit_Result
0.224
'''
##############
# Question 8 #
##############
'''
Assume that if your model gives a .5 probability or greater, then it predicts that fraud occurred; if it predicts a probability less than .5, then it predicts that fraud did not occur.

What is the ratio of correct predictions to the total number of observations used for the logit model?
'''

Logit_predicted_poss = Logit_Result.predict()
Logit_predicted_outcomes = []

for x in Logit_predicted_poss:
    if x >= 0.5:
        Logit_predicted_outcomes.append(1)
    else:
        Logit_predicted_outcomes.append(0)   
        
logit_hits = 0
logit_i = 0

for fraud_q8 in Y:
    if Logit_predicted_outcomes[logit_i] == fraud_q8:
        logit_hits = logit_hits + 1
        
    logit_i = logit_i + 1
    
Logit_precent_correct = logit_hits/len(Y)
print('Q8: ', round(Logit_precent_correct, 3)) #0.859 - Correct


##############
# Question 9 #
##############

predict_y = Logit_Result.predict(X)
ax = sns.lineplot(x=df['v.2months.pc'], y=predict_y, label='Logit Model', color='r')
ax.set(xlabel='v.2months.pc', ylabel='Fraud')
#plt.show()