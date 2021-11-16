import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import missingno as msno
import seaborn as sns
import statistics
import scipy.stats as stats
from scipy.stats import skew
import statsmodels.formula.api as sm

pd.options.display.float_format = '{:.2f}'.format
#pd.options.display.width = None
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None



'''

Build models using OLS regression to predict sale price
Give measurements of the qualities of the models you build
At every step, explain the assumptions, limitations, and ramifications of what you are doing. At the end, include a summary of what you did and what you found

- random sample of homes that have sold from three regions (Northern Township, City of Chicago and Southwest Township).
- during tax years of 2002 - 2018
- 20000 total records


	
Areas of Grading:
	- Executive Summary - 10 points: briefly introduce the motivation, data, methods, and results. Be concise.
	- Introduction (Motivation) - 10 points: the purpose of this project. Feel free to use the context we provided.
	- Dataset Description - 10 points: describe the dataset in terms of source, dimension, size, and other things you might find relevant. Feel free to use the context we provided.
	- Data Cleaning - 10 points: address issues like data type conversion, fixed structure for text data, missing values, and outliers.
	- Exploratory Data Analysis (EDA) - 10 points: data visualizations of distributions, relationships, and any other information that has certain insights. Please include at least 3 nicely formatted plots (wth title, x- and y-axis label and legend if applicable) and your interpretations of them.
	- Correlation Analysis and Hypothesis Testing - 10 points: please focus on the relationship between the dependent variable and the predictors. Please include a correlation matrix (Pearson, Spearman, or Kendall) and at least 2 unique types of statistical tests (t-test, two-sample t-test, ANOVA, normal distribution tests, etc.). Please include interpretations of the results for each.
	- Feature Selection, Engineering, and Preprocessing - 10 points: exclude existing features and create new features. Transform the data to numerical and prepare it for subsequent regression analyses. Change the unit of the feature, apply nonlinear transformations (take natural logarithms), and standardize the feature if you see fit.
	- Modeling and Optimization - 20 points: build linear regression models (single or multiple) and try to optimize and find the one with the highest (adjusted) R
	- Model Diagnostics - optional, 10 points extra: test the assumptions made by your "best model". Refer to external resources to learn how.
	- Conclusion - 10 points:
'''

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/project1data.csv')




#EXECUTIVE SUMMARY
'''
Using Cook Counties organic sample data from the following regions (Northern Township, City of Chicago and Southwest Township), the scope of this assessment is to provide an unbiased analytical assessment that will provide an Cook County Tax Assessors an alternative way forward with regards to how property taxes are assessed in a fair manner to not only the County but also the tax payers.
'''


#INTRODUCTION
'''
The purpose of this project is to address the taxation issues presented by Rob Ross are as follows:
	- determine valuation gap between pre and post appeals (where did initial valuation go wrong?)
	- explore data (remove outliers: visualize data (poss. look at log to normalize data/graphs); create composite/joint values
	- create graphs to support research
	- conduct test to determine which variables are worth exploring
	- build OLS model to predict sales price
	- give measurements of the models
	
'''


#DATASET DESCRIPTION
'''
The dataset comes from a random sample of homes that have sold from three regions (Northern Township, City of Chicago and Southwest Township).
The date range for the tax data is from: 2002 - 2018
There are 20000 records in this dataset.
'''

def dataset():
	global df
	print(df.info()) # (20000, 21)
	print(df.shape)
	print(df.size) # output = rows * columns  OR output = df.shape[0] * df.shape[1] (420000)
	print(df.dtypes) #items to note here that some of the column categories are not integers and/or floats.  
	print(df.isna().sum()) #groups columns and displays a sum of missing values.
	print(df.isna().any()) #confirms above (isna().sum() by returning boolean value.
	print(df.groupby('Tax_Year').count()) #ensure no data outside of the mandated 2002-2018 time frame.
	print(df['Property_Address'].is_unique) #False - there are duplicate property addresses
	print(df[df.duplicated(['Property_Address'])].sort_values(['Property_Address']))
	print(df['Property_Address'].drop_duplicates())
	df = df.drop_duplicates(subset='Property_Address') #798 rows dropped
	print(df['Property_Address'].is_unique) #True - there are no duplicate property addresses

#dataset()
print('df dataset: ', df.shape[0])
#DATA CLEANING

def data_cleaning():
	global df
	df_sale_filtered = df[df['Sale_Price'] < 10].index #OUTLIERS: total homes with a sale price of 1 (count of 348)
	df = df.drop(df_sale_filtered)
	print(df.shape) # new column count of 19652 after the sale price 1 homes dropped

	print(msno.matrix(df, figsize=(11,6), fontsize=10))
	#plt.show()

#data_cleaning()
print('df data cleaning: ', df.shape[0])

'''
Review Property Classes and create new column (regression class/non-regression class) based on field value.
Remove rows with Sales Price less than or equal to 1.
drop all fields that are NA
Change categories of fields (non int/float) that will not allow for regression analysis.
'''

#EXPLORATORY DATA ANALYSIS

def eda():
	print(df.sample(10))
	print(df.describe())
	print(df['Sale_Price'].skew)

	fig, axes = plt.subplots(1, 3, figsize=(11,8))

	sns.distplot(df['Sale_Price'], kde=False, color='r', ax=axes[0])
	sns.distplot(df.groupby(['Property_Class']).count(), hist=False, color='b', ax=axes[1], label='Property Class')
	sns.distplot(df['Walkscore'], color='g', ax=axes[2], label='Walkscore')
	sns.relplot(data=df, x='Rooms', y='Sale_Price')

	plt.show()
	
#eda()
print('df eda: ', df.shape[0])

#CORRELATION ANALYSIS AND HYPOTHESIS
'''
build a heat map
'''
corr_matrix = df.corr #correlation matrix
fig, axes = plt.subplots(figsize=(11,8))
sns.heatmap(round(corr_matrix(method='pearson'), 2), ax=axes, annot=True, square=True) #pearson - input raw data for linear relationship
fig.subplots_adjust(top=0.9)
fig.suptitle('Correlation Matrix')
'''
Correlation Matrix (positively correlated): Rooms/Bedrooms = .88 ; Rooms/Full Baths = .74; Rooms/BLDG Sq Feet = .77; Full Baths/BLDG Sq Feet = .8
'''

#numeric = df.dtypes[df.dtypes != 'object'].index
#skewed = df[numeric].apply(lambda x: skew(x.dropna()))
#skewed = skewed[skewed > 0.5]
#skewed = skewed.index
#print('Number of skewed features: ', len(df[skewed].columns.values))
#
#df[df[skewed].columns.values].hist(bins=50, figsize=(10,10))
#plt.show()


df_bedrooms = list(df['Bedrooms'].dropna())
df_rooms = list(df['Rooms'].dropna())
print(stats.ttest_ind(df_rooms, df_bedrooms, equal_var=False)) #Ttest_indResult(statistic=122.34734263305134, pvalue=0.0)
'''
NOTE: equal_var=False because it is assumed that the population variance is not equal (see variance and mean results below)
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
'''
print(statistics.variance(df_rooms), statistics.variance(df_bedrooms)) #9.432302503866596 1.9413244702331345
print(statistics.mean(df_rooms), statistics.mean(df_bedrooms)) #6.948621908127208 3.479929328621908

'''
TO DO: t test and Normal Dist test.
'''


#FEATURE SELECTION, ENGINEERING, AND PREPROCESSING
'''
remove/add features
take non-numerical data and make it numerical
apply log to feature 

'''
#remove/add features
df.insert(20, 'Walkfac_Int', '')
df_walkfac_list = list(df.Walkfac.unique()) #['Car-Dependent', 'Somewhat Walkable', 'Very Walkable', "Walker's Paradise"]
for x in range(0,4):
	df.loc[df['Walkfac'] == df_walkfac_list[x], 'Walkfac_Int'] = x+1 #udpates the value of Walkfac_Int to an int, based on Walkfac.
	
df['Walkfac_Int'] = df['Walkfac_Int'].astype('int64') #use of .astype method to change dtype



# MODELING AND OPTIMIZATION
print(df.info())
model = sm.ols(formula='Rooms ~ Bedrooms + Full_Baths + Building_Square_Feet', data=df)
result = model.fit()
print(result.summary())

#MODEL DIAG:
	
