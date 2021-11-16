import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as sm
import statsmodels.api as sm_api

from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import stats

pd.options.display.max_columns = None

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 2/project1data.csv')

df = df.dropna() #drop all fields that are NA
prop_list = (202, 203, 204, 205, 206, 207, 208, 209, 210, 234, 278, 295) # set property class to the following fields
df = df[df['Property_Class'].isin(prop_list)] # Apply this property class to filter only the above fields
df = df[df['Building_Square_Feet'] > 0] #BSF was filtered out any fields less than zero.  These zero fields were kicking errors when applying log in the plt.figure

#Graphical representation of the below fields and what the distributions look like pre/post log.
fig, axes = plt.subplots(2, 3, figsize=(12,8), dpi=100)
#Sale_Price, Building_Square_Feet, and Land_Square_Feet
sns.histplot(df, x='Sale_Price', kde=False, color='r', ax=axes[0,0])
axes[0,0].set_title('Sale Price')

sns.histplot(df, x='Building_Square_Feet', kde=False, color='b', ax=axes[0,1])
axes[0,0].set_title('Building_Square_Feet')

sns.histplot(df, x='Land_Square_Feet', kde=False, color='g', ax=axes[0,2])
axes[0,0].set_title('Land_Square_Feet')

sns.histplot(df, x='Sale_Price', stat='count',  kde=False, color='r', log_scale=True, ax=axes[1,0])
axes[0,0].set_title('Sale Price - Log')

sns.histplot(df, x='Building_Square_Feet', stat='count', kde=False, color='b', log_scale=True, ax=axes[1,1])
axes[0,0].set_title('Building_Square_Feet - Log')

sns.histplot(df, x='Land_Square_Feet', stat='count', kde=False, color='g', log_scale=True, ax=axes[1,2])
axes[0,0].set_title('Land_Square_Feet - Log')
#	plt.show()

#	Insert the following columns, leaving the values blank to be filled in below
df.insert(21, 'Log_Sale_Price', '')
df.insert(22, 'Log_Building_Square_Feet', '')
df.insert(23, 'Log_Land_Square_Feet', '')


#	Set the newly created columns to the np.log of the value of it's mirrored column
df['Log_Sale_Price'] = np.log(df['Sale_Price'])
df['Log_Building_Square_Feet'] = np.log(df['Building_Square_Feet'])
df['Log_Land_Square_Feet'] = np.log(df['Land_Square_Feet'])

#	Set df as any values greater than 0
df = df[df['Log_Sale_Price'] > 0]
df = df[df['Log_Building_Square_Feet'] > 0]
df = df[df['Log_Land_Square_Feet'] > 0]

print('DF SHAPE: ', df.shape)


df.insert(16, 'Age_Binned', '')
df_age_binned_list = ["+2006", "2005-1976", "1975-1946", "1945-"]
df.loc[df['Age'] <= 13, 'Age_Binned'] = df_age_binned_list[0]
df.loc[(df['Age'] >= 14) & (df['Age'] <= 43), 'Age_Binned'] = df_age_binned_list[1]
df.loc[(df['Age'] >= 44) & (df['Age'] <= 73), 'Age_Binned'] = df_age_binned_list[2]
df.loc[df['Age'] >= 74, 'Age_Binned'] = df_age_binned_list[3]



############
#QUESITON 1#
############
print('Question 1: ', df[df['Age_Binned'] == '2005-1976'].shape[0])
#print('bin 2006+: ', df[df['Age_Binned'] == '+2006'].shape[0]) #used to confirm shape
#print('bin 1975-1946: ', df[df['Age_Binned'] == '1975-1946'].shape[0]) #used to confirm shape
#print('bin 1945-: ', df[df['Age_Binned'] == '1945-'].shape[0]) #used to confirm shape
	
############
#QUESITON 2#
############

print('Question 2: ', np.mean(df['Log_Land_Square_Feet']))

############
#QUESITON 3#
############

#	print(df['Log_Sale_Price'].shape) #used to keep track of changes to dataframe shape count
df.sort_values(by=['Log_Sale_Price'], inplace=True)
q1, q3 = np.percentile(df['Log_Sale_Price'],[25,75])
iqr = q3 - q1
LSP_q1 = q1 - 1.5 * iqr
LSP_q3 = q3 + 1.5 * iqr


df = df[df['Log_Sale_Price'] <= LSP_q3]
df = df[df['Log_Sale_Price'] >= LSP_q1]
#	print(df_new['Log_Sale_Price'].shape) #used to keep track of changes to dataframe shape count
lsp_mean = np.mean(df['Log_Sale_Price'])
print('Question 3: ', round(lsp_mean, 3))

print('DF SHAPE ', df.shape)
df_ab = df['Age_Binned'].value_counts()
print('DF Age Binned (to_string): ', df_ab.to_string())

###################
#QUESTION 4, 5 & 6#
###################

'''
leave out +2006 (Ages_Binned) and Car_Dependent (Walkfac)

Log_Building_Square_Feet
Log_Land_Square_Feet
Somewhat_Walkable, Very_Walkable, Walkers_Paradise
2005-1976, 1975-1946, 1945

'''
ab_dummy = pd.get_dummies(df['Age_Binned'])
ab_dummy.columns = ['_2006_Up','_1945_Down','_1975_1946_','_2005_1976_']

walkfac_dummy = pd.get_dummies(df['Walkfac'])
walkfac_dummy.columns =['Car_Dependent','Somewhat_Walkable','Very_Walkable','Walkers_Paradise']
print(walkfac_dummy)

df = pd.concat([df, ab_dummy], axis=1)
df = pd.concat([df, walkfac_dummy], axis = 1)

#print(df.sample(10))
#	print(ab_dummy.iloc[:, 0:5]) #prints selected columns: 2 - 4 (1945-  1975-1946  2005-1976)
#	print(walkfac_dummy.iloc[:, 0:4]) #prints selected columns: 1 - 3 (Somewhat Walkable  Very Walkable  Walker's Paradise)

model = sm.ols(formula='Log_Sale_Price ~ Log_Building_Square_Feet + Log_Land_Square_Feet + Somewhat_Walkable + Very_Walkable + Walkers_Paradise + _2005_1976_ + _1975_1946_ + _1945_Down ', data=df)
result = model.fit()
print('Question 4: ', result.summary())

############
#QUESTION 7#
############

ind_variables = df[['Log_Building_Square_Feet', 'Log_Land_Square_Feet', 'Somewhat_Walkable', 'Very_Walkable', 'Walkers_Paradise', '_2005_1976_', '_1975_1946_', '_1945_Down' ]]
ind_variables.insert(0, 'const', 1) #insert a column, named "Const", at the zero column position filled with ones.
VIF_values = []
for i in range(len(ind_variables.columns)): #Loop through the dataframe and calculate VIF for each column
	VIF = variance_inflation_factor(ind_variables.values, i) #calculate the VIF for the matrix on the 'ith' column.
	VIF_values.append(VIF)
VIFs = pd.DataFrame({'VIF':VIF_values}, index=ind_variables.columns)
print('Question 7 (Variation Inflation Factor (VIF): \n', round(VIFs, 3))


############
#QUESTION 8#
############
'''
df_2006_vc = df['_2006_Up'].value_counts()
print('df 2006: \n', df_2006_vc.to_string()) # by using to_string ( or .values), the 'output' and 'dtype' objects in the print output

df_2005_1975_vc = df['_2005_1976_'].value_counts()
print('df 2005 - 1975: \n', df_2005_1975_vc.to_string())

df_1975_1946_vc = df['_1975_1946_'].value_counts()
print('df 1975 - 1945: \n', df_1975_1946_vc.to_string())

df_1945_vc = df['_1945_Down'].value_counts()
print('df 1945: \n', df_1945_vc.to_string())
'''


model_q8_unrestricted = sm.ols(formula = 'Log_Sale_Price ~ Log_Building_Square_Feet + Log_Land_Square_Feet + Somewhat_Walkable + Very_Walkable + Walkers_Paradise + _2005_1976_ + _1975_1946_ + _1945_Down', data=df)
unrestrict_result = model_q8_unrestricted.fit()

ssr_ur = sum(unrestrict_result.resid**2) #sum squared residuals - ur

model_q8_restricted = sm.ols(formula='Log_Sale_Price ~ Log_Building_Square_Feet + Log_Land_Square_Feet + Somewhat_Walkable + Very_Walkable + Walkers_Paradise', data=df)
restrict_result = model_q8_restricted.fit()
print('Question 8: ', round(restrict_result.ssr, 3)) 
#Result of 3648.899 is incorrect
#Result of 2218.763 is correct
#NOTE: To properly execute this restricted model, the fields that were testing against MUST BE REMOVED from the variable list.

ssr_r = sum(restrict_result.resid**2) #ssr_r = sum of squared residuals - restricted model (RESTRICTED means not all variable (AKA features) included in model
print('Question 8 (alt ssr_r): ', round(ssr_r, 3))


#An alternate solution is listed below.  This model was built and fit using statsmodel.api
X = df[['Log_Building_Square_Feet','Log_Land_Square_Feet','Somewhat_Walkable','Very_Walkable','Walkers_Paradise']] # X is set to the independent variables.
X = sm_api.add_constant(X) #set the constant term manually with statsmodel api
y = df['Log_Sale_Price'] # y is the dependent variable
model_q8 = sm_api.OLS(y, X) #set the model_q8 object from the Ordinary Least Squares
result_q8 = model_q8.fit()
print('Question 8 (alt ssr): ', round(result_q8.ssr, 3))


#An alternate solution is listed below.  This model was built and fit using sklearn.
model_sklearn_q8 = LinearRegression()
model_sklearn_q8.fit(X,y)
yhat = model_sklearn_q8.predict(X)
SS_Residual_q8 = sum((y-yhat)**2)
print('SKLearn SSR - q8: ', round(SS_Residual_q8, 3))
#print(dir(model_sklearn_q8))


#############
#QUESTION 9#
#############
k = 8 #number of variables in the unrestricted model (model_q8_unrestricted)
q = 3 #number of equals signs in the restriction
n = len(df)


f_stat = ((ssr_r - ssr_ur)/q)/(ssr_ur/(n-k-1))
print('Q9: ', round(f_stat, 3))
print('Question 9 (F Statistic): ', round(restrict_result.fvalue, 3))
#NOTE: res.fvalue is not the same as F Statistic.  
#F Statistic is not an attribute of a res model.  
#F Statistic must be calculate manually with statsmodels
print('Question 9 (P Value): ', restrict_result.f_pvalue) 
# this was ran to check the "f_pvalue" arguement of res and compare to the summary output in question 10 (CHECKED OUT)
#print('Question 9: ', stats.f_oneway(df['A_2005_1976_B'], df['A_1975_1946_B'], df['A_1945_B']))

#############
#QUESTION 10#
#############
print('Question 10: ', restrict_result.summary())
