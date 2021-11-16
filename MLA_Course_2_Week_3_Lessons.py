import pandas as pd
import numpy as np
import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import logistic



######################################
#LINEAR PROBABILITY MODEL (First Lab)#
### BINARY CLASSIFICATION WITH LPM ###
######################################
df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/Week_3_Astan_Violence_replicationdata.csv')
#print(df.info())
df = df.dropna()

def LPM():
    global df
    X = df['v.elday.pc']
    Y = df['fraudchi2_total']
    LPM.LPM_Model = sm.OLS(Y,X)
    LPM.LPM_Result = LPM.LPM_Model.fit()
    print(LPM.LPM_Result.summary())
    LPM.predict_y = LPM.LPM_Result.predict(X) #generate the model's prediction
    ax = sns.scatterplot(data=df, x=df['v.elday.pc'], y=df['fraudchi2_total']) #graph the original data
    sns.lineplot(x=X, y=LPM.predict_y, label='LPM', color='r') #graph the model as a line
    
    #return LPM_Result, predict_y
    plt.show()
LPM()


#consequnce of LPM, is that R-Squared and R-Squared ADJ are no longer useful measurements if the independent variables are continous.

def LPM_Consequence():
    squared_residuals = LPM.LPM_Result.resid**2
    ax1 = sns.scatterplot(x=LPM.predict_y, y=squared_residuals)
    ax1.set(xlabel='Predictions', ylabel='Squared Predictions')
    plt.show()
#LPM_Consequence()
    
    
def LPM_Fraud_Model():
    global df   
    X = df[['v.elday.pc','v.elday.pc.2','numclosedstations','electrified','pce_1000_07','distkabul','elevation']]
    y = df['fraudchi2_total']
    X = sm.add_constant(X)
    LPM_Model1 = sm.OLS(y, X)
    LPM_Result1 = LPM_Model1.fit()
    predY = LPM_Result1.predict(X)
    ax2 = sns.lineplot(data=df, x='v.elday.pc', y=predY)
    ax2.set(xlabel='v.elday.pc', ylabel='Predicted Prob of Fraud') 
    
     
    plt.show()
#LPM_Fraud_Model()


###################################
# BINARY CLASSIFICATION W/ PROBIT #
###################################

df1 = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/weidmannandcallen.csv')
df1 = df1.dropna()

def Probit_Model_Func():
    X = df[["v.elday.pc", "v.elday.pc.2", "numclosedstations", "electrified", "pce_1000_07", "distkabul", "elevation"]]
    y = df["fraudchi2_total"]
    Probit_Model_Func.X = sm.add_constant(X)
    Probit_Model = sm.Probit(y, Probit_Model_Func.X)
    Probit_Model_Func.Probit_Result = Probit_Model.fit()
    predY = Probit_Model_Func.Probit_Result.predict(Probit_Model_Func.X)
    ax = sns.lineplot(data=df, x='v.elday.pc', y=predY, label='Probit Model', color='r')
    sns.scatterplot(data=df, x='v.elday.pc', y=y)
    ax.set(xlabel='v.elday.pc', ylabel='Fraud')
    plt.show()
#Probit_Model_Func()

def PEA():
    #PEA = Partial Effect at the Average
    
    #The partial derivative with respect to v.elday will be its coefficient times the pdf of the linear function
    #First, we need to calculate the linear function with all variables set to their mean
    #This is a lot easier with matrices! See below.
    linear_model = Probit_Model_Func.Probit_Result.params[0] + Probit_Model_Func.Probit_Result.params[1]*df["v.elday.pc"].mean() + Probit_Model_Func.Probit_Result.params[2]*df["v.elday.pc.2"].mean() + Probit_Model_Func.Probit_Result.params[3]*df["numclosedstations"].mean() + Probit_Model_Func.Probit_Result.params[4]*df["electrified"].mean() + Probit_Model_Func.Probit_Result.params[5]*df["pce_1000_07"].mean() + Probit_Model_Func.Probit_Result.params[6]*df["distkabul"].mean() + Probit_Model_Func.Probit_Result.params[7]*df["elevation"].mean()
    #Then, we take the pdf and multiply it by the coefficient of v.elday.pc, which is params[1] 
    PEA = norm.pdf(linear_model, 0, 1)*Probit_Model_Func.Probit_Result.params[1]
    print("The partial effect at the average for v.elday.pc is %f" % PEA)
#PEA()

def APE():
    #APE = Average Partial Effect
    
    #First, we calculate the dot product of the values for X and the ceofficients in our model
    linear_models = Probit_Model_Func.X.dot(Probit_Model_Func.Probit_Result.params)
    #Then, we calculate their pdfs
    pdfs = norm.pdf(linear_models)
    #Next, we multiply the pdfs by the coefficient for v.elday.pc
    partial_effects = pdfs*Probit_Model_Func.Probit_Result.params[1]
    #Finally, we take the average
    APE = partial_effects.mean()
    print("The APE for v.elday.pc is %f" % APE)
#APE()
    
    
###################################
# BINARY CLASSIFICATION W/ LOGIT #
###################################
def logit():
    logistic_x = np.linspace(logistic.ppf(0.01), logistic.ppf(0.99), 1000)
    logistic_pdf = logistic.pdf(logistic_x)
    logistic_cdf = logistic.cdf(logistic_x)
    
    norm_x = np.linspace(norm.ppf(0.01), norm.ppf(0.99), 1000)
    norm_pdf = norm.pdf(norm_x)
    norm_cdf = norm.cdf(norm_x)
    
    fig, ax = plt.subplots(2,2)
    fig.tight_layout()
    ax[0,0].plot(logistic_pdf)
    ax[0,0].title.set_text("Logistic Distribution's PDF")
    ax[1,0].plot(logistic_cdf)
    ax[1,0].title.set_text("Logistic Distribution's CDF")
    ax[0,1].plot(norm_pdf)
    ax[0,1].title.set_text("Normal Distribution's CDF")
    ax[1,1].plot(norm_cdf)
    ax[1,1].title.set_text("Normal Distribution's CDF")
    plt.show()
#logit()

def logit_practice():
    
    df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/weidmannandcallen.csv')
    df = df.dropna()
    X = df[["v.elday.pc", "v.elday.pc.2", "numclosedstations", "electrified", "pce_1000_07", "distkabul", "elevation"]]
    y = df['fraudchi2_total']
    X = sm.add_constant(X)
    LModel = sm.Logit(y,X)
    LResult = LModel.fit()
    predY = LResult.predict(X)
    
    ax = sns.lineplot(x=df['v.elday.pc'], y=predY, label='Logit Model', color='r')
    sns.scatterplot(x=df['v.elday.pc'], y=y)
    ax.set(xlabel='v.elday.pc', ylabel='Fraud')
      
    
    #######################################
    # Partial Effect at the Average (PAE) #
    #######################################
    #The partial derivative with respect to v.elday will be its coefficient times the pdf of the linear function
    #First, we need to calculate the linear function with all variables set to their mean
    #This is a lot easier with matrices! See below.
    linear_model = LResult.params[0] + LResult.params[1]*df["v.elday.pc"].mean() + LResult.params[2]*df["v.elday.pc.2"].mean() + LResult.params[3]*df["numclosedstations"].mean() + LResult.params[4]*df["electrified"].mean() + LResult.params[5]*df["pce_1000_07"].mean() + LResult.params[6]*df["distkabul"].mean() + LResult.params[7]*df["elevation"].mean()
    #Then, we take the pdf and multiply it by the coefficient of v.elday.pc, which is params[1] 
    Logit_PEA = logistic.pdf(linear_model, 0, 1)*LResult.params[1]
    print("The partial effect at the average for v.elday.pc is %f" % Logit_PEA)  
    
    ################################
    # Average Partial Effect (APE) #
    ################################
    #First, we calculate the dot product of the values for X and the ceofficients in our model
    linear_models = X.dot(LResult.params)
    #Then, we calculate their pdfs
    pdfs = logistic.pdf(linear_models)
    #Next, we multiply the pdfs by the coefficient for v.elday.pc
    logit_partial_effects = pdfs*LResult.params[1]
    #Finally, we take the average
    logit_APE = logit_partial_effects.mean()
    print("The APE for v.elday.pc is %f" % logit_APE)
    
    ##########################################
    # Compare partial effects for v.elday.pc #
    ##########################################
    LPMModel = sm.OLS(y,X)
    LPMResult = LPMModel.fit()
    LPM_APE = LPMResult.params[1] 
    PModel = sm.Probit(y,X)
    PResult = PModel.fit()
    probit_pdfs = norm.pdf(X.dot(PResult.params))
    probit_partial_effects = probit_pdfs*PResult.params[1]
    probit_APE = probit_partial_effects.mean()
    print("The partial effect for the LPM is %f; the APE for the probit model is %f; the APE for the logit model is %f" % (LPM_APE, probit_APE, logit_APE))    
        
    #################################
    # Graph Probit and Logit Models #
    #################################
    LPM_pred = LPMResult.predict(X)
    probit_pred = PResult.predict(X)
    logit_pred = LResult.predict(X)
    sns.lineplot(x=df["v.elday.pc"], y=LPM_pred, label="LPM")
    sns.lineplot(x=df["v.elday.pc"], y=probit_pred, label="Probit")
    sns.lineplot(x=df["v.elday.pc"], y=logit_pred, label="Logit") 
    plt.show()
  
#logit_practice()



