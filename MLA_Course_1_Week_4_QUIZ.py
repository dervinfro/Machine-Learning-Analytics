import pandas as pd
import numpy as np
import statistics
import scipy.stats as stats
from scipy.stats import chi2_contingency
import time

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/quiz4prices.csv')

def question_one():
	'''
	Question 1:
	Intuitively, you suppose that central air will be an important factor to take into account in valuating properties. But you need to conduct testing to confirm. Your plan is to use a t-test, meaning that you will need the samples' means and standard deviations.

	For this sample, what is the standard deviation for the sale price for properties without central air? (round to two decimal places; don't include commas)
	'''
	global df
	air_groupby = df.groupby('Central_Air').count() #2789 = No // 4181 == Yes
	question_one.df_air_no = df[df['Central_Air'] == 'No']
	question_one.df_sp_no = question_one.df_air_no['Sale_Price'] #dataframe sale price - no air
	sp_stdev = statistics.stdev(question_one.df_sp_no) #sale price standard deviation
	print('Q1 -- sales price stdev: ', round(sp_stdev, 2)) #205460.3
	
	return air_groupby

print(question_one())

def question_two():
	'''
	Question 2:
	What is the sample average sale price for properties with no central air? (round to two decimal places; do not include commas)
	'''
	sp_airno_mean = statistics.mean(question_one.df_sp_no) #sale price - no air mean
	print('Q2 -- sales price no air - mean: ', round(sp_airno_mean, 2)) #265950.36

print(question_two())


def question_three():
	'''
	Question 3:
	Conduct a two sample t-test on sale price for properties with and without central air and report the t-statistic. (round to two decimal places)
	'''
	df_sp_air_no = df_air_no['Sale_Price'] #dataframe sale price air no (df_air_no <- Q1)
	df_air_yes = df[df['Central_Air'] == 'Yes']
	df_sp_air_yes = df_air_yes['Sale_Price'] #dataframe sale price air yes
	print('Q3 -- t-test results (AKA: t-statistics): ', stats.ttest_ind(df_sp_air_no, df_sp_air_yes)) #mean: 336313.56 #Ttest_1sampResult(statistic=20.36848629439786, pvalue=5.23372718252841e-88)
		#Ttest_1sampResult(statistic=5.141441250638327e-07, pvalue=0.9999995897870565) (UPDATED)
		#Ttest_indResult(statistic=-15.166656444203158, pvalue=3.8296057969069295e-51) (UDPATED - V2)


def question_four():
	'''
	Question 4:
	You also want to understand whether the Walkfac is statistically significant for Sale Price--that is, whether a property is coded as Car-Dependent, Somewhat Walkable, Very Walkable, or Walker's Paradise based on its Walkscore. You will conduct an ANOVA test for this purpose, which means comparing the variance between groups to the variance within groups.

	Calculate the residual sum of squares between groups (SSB). (round to the nearest whole number)

	'''
	#print(df.groupby('Walkfac').count())
	car_dep = df[df['Walkfac'] == 'Car-Dependent'] # 2648 occurances
	some_walk = df[df['Walkfac'] == 'Somewhat Walkable'] #2120 occurances
	very_walk = df[df['Walkfac'] == 'Very Walkable'] #1868 occurances
	walk_paradise = df[df['Walkfac'] == "Walker's Paradise"] #334 occurances

	total_sp_mean = statistics.mean(df['Sale_Price']) #total sale price mean = 336313.56

	walk_list = [car_dep, some_walk, very_walk, walk_paradise]

	mean = [335828.68919939577, 299005.96745283017, 321118.39293361886, 661944.1586826347]
	number = [2648, 2120, 1868, 334]

	ssb = 0
	for x in walk_list:
		ssb = ssb + (len(x) * (x['Sale_Price'].mean() - total_sp_mean)**2)
	print('Q4 -- Sum of Square between Groups: ', ssb) #number of groups - 1
		#12932817449417 (UPDATED)
		#38798452348251.58 (UPDATED - V2)

def question_five():
	'''
	Question 5:
	Calculate the residual sum of squares within groups (SSW). (round to the nearest whole number)
	'''

	start_time = time.time()

	list_ssw = []
	ssw = 0
	for x in walk_list:
		for y in x['Sale_Price']:
	#		ssw = ssw + (y - x['Sale_Price'].mean()) ** 2
			list_ssw.append((y - x['Sale_Price'].mean()) ** 2) #this is an alternate method 
	#print('Q5 -- sum of squares within groups: ', ssw) #681522589945599.2
	print('Q5 -- sum of squares using list append: ', sum(list_ssw))
	end_time = time.time()
	print('Time Loop: {}'.format(end_time - start_time))


def question_six():
	'''
	Question 6:
	Calculate the f-statistic. (round to two decimal places)
	'''
	print('Q6 -- ANOVA F Oneway: ', stats.f_oneway(car_dep['Sale_Price'], some_walk['Sale_Price'], very_walk['Sale_Price'], walk_paradise['Sale_Price'])) #statistic=132.1893179796605, pvalue=2.7449315507302748e-83 (UPDATED)


def question_seven():
	'''
	Question 7:
	As you know, conducting a chi-squared test involves comparing the observed frequencies and expected frequencies. What is the expected frequency for properties that ARE occupied by the homeowner where an appeal IS filed? (round to the nearest whole number)
	'''
	df1 = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/quiz4appealsv2.csv')

	contingency_table = pd.crosstab(df1['homeowner'], df1['Did_Appeal'], margins=True)

	print(contingency_table.shape)
	chi2, p, dof, expected = chi2_contingency(contingency_table)
	print('Q7 -- expected: \n', expected) #996


def question_eight():
	'''
	Question 8:
	What is the chi-squared statistic for this relationship? (round to two decimal places)
	'''
	print('Q8 -- Chi2: ', chi2) #6.678780614743408


def question_nine():
	'''
	Question 9:
	What is the p-value corresponding to the chi-squared statistic? (round to two decimal places)
	'''
	print('Q9 -- p value: ', p) #0.15386857480303526
	
def test_question():
	
	air_groupby = question_one() #call a variable that is inside another function
	test = question_one.df_air_no

#	global df
#	
#	for row in df.itertuples():
#		print(row.Sale_Price)
		
	return air_groupby, test

#print(test_question())