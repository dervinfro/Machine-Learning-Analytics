import pandas as pd
import numpy as np
import statistics

from scipy.stats import skew
from scipy.stats import kurtosis
from scipy import stats

#pd.set_option('display.max_rows', None)
#pd.set_option('display.max_columns', None)

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment2.csv')
df_sp = df['Sale_Price'] #dataframe sale price


def question_1():
	df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment2.csv')
	df_sp = df['Sale_Price'] #dataframe sale price
	
	df_q1_list = []
	
	df_sp_mean = df_sp.mean() #dataframe sale price mean
	stat_mean = statistics.mean(df_sp) #alt example for mean

	df_sp_median = df_sp.median() #dataframe sale price median
	stat_median = statistics.median(df_sp) #alt example for median

	df_sp_variance = df_sp.var() # squared deviation of a variable from its mean.  Low value = data is not spread apart widely // High value = data is spread apart widely.
	stat_var = statistics.variance(df_sp) #alt example for variance

	df_q1_list.extend([df_sp_mean, df_sp_median, df_sp_variance])
	answer_1 = df_q1_list
	
	return(answer_1)
print(question_1())

def question_2():
	df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment2.csv')
	df_sp = df['Sale_Price'] #dataframe sale price
	
	df_q2_list = []	
	
	#measure skew correcting for bias
	df_sp_skew = skew(df_sp, bias=False)
	
	#measure kurtosis, correcting for bias (and do NOT give excess kurtosis)
	df_sp_kurt = kurtosis(df_sp, bias=False, fisher=False)
	
	df_q2_list.extend([df_sp_skew, df_sp_kurt])
	
	answer_2 = df_q2_list
	
	return(answer_2)
print(question_2())

def question_3():
	df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment2.csv')
	df_sp = df['Sale_Price'] #dataframe sale price
	
	np_q3_list = [] #numpy question 3 list
	
	#calculate IQR
	q3_iqr = stats.iqr(df_sp, interpolation='midpoint') #question 3 iqr 201000
	q3_iqr_qtr1 = np.percentile(df_sp, 25, interpolation='midpoint' ) #154000
	q3_iqr_qtr3 = np.percentile(df_sp, 75, interpolation='midpoint') #355000
	np_q3_list.extend([q3_iqr_qtr1, q3_iqr_qtr3])
	
	#calculate upper and lower cut offs
	q3_iqr_qtr1 = np.percentile(df_sp, 25, interpolation='midpoint' ) #question 3 - IQR - quarter 1
	q3_iqr_qtr3 = np.percentile(df_sp, 75, interpolation='midpoint') #question 3 - IQR - quarter 3
		
	q3_iqr_lower = q3_iqr_qtr1 - (1.5 * q3_iqr) #question 3 iqr lower
	q3_iqr_upper = q3_iqr_qtr3 + (1.5 * q3_iqr)	#question 3 iqr upper
	
	print(q3_iqr_qtr1, q3_iqr_qtr3, q3_iqr_lower, q3_iqr_upper)
	
	#remove data that falls above or below the cut off
	df = df[df_sp < q3_iqr_upper] #dataframe that is less than iqr upper
	
	answer_3 = df
	
	return(answer_3) #return the dataframe without the outliers
#print(question_3())	

def question_4():
	df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment2.csv')
	df_sp = df['Sale_Price'] #dataframe sale price
	
	df_q4_list = []
	
	# Create a new series that is the log of the values in Sale_Price
	df_log = np.log(df_sp)
	
	# Calculate IQR
	q4_iqr = stats.iqr(df_log, interpolation='midpoint') #question 4 iqr -- value: 0.8351651870617864

	# Calculate upper and lower cut-offs
	q4_iqr_qtr1 = np.percentile(df_log, 25, interpolation='midpoint') #question 4 iqr quarter 1
	q4_iqr_qtr3 = np.percentile(df_log, 75, interpolation='midpoint') #question 4 iqr quarter 3
	
	q4_iqr_lower = q4_iqr_qtr1 - (1.5 * q4_iqr)
	q4_iqr_upper = q4_iqr_qtr3 + (1.5 * q4_iqr)
	
	print(q4_iqr_qtr1, q4_iqr_qtr3, q4_iqr_lower, q4_iqr_upper)
	
	# Remove data that falls below or above the cut off
	df = df[(df_log < q4_iqr_upper) & (df_log > q4_iqr_lower)] #np.log of df['Sale_Price'] is: 9566

	# Calculate the skew and kurtosis
	df_log_skew = skew(np.log(df['Sale_Price']), bias=False)
	
	df_log_kurtosis = kurtosis(np.log(df['Sale_Price']), bias=False, fisher=False)
	
	df_q4_list.extend([df_log_skew, df_log_kurtosis])
	
	answer_4 = df_q4_list #9566 is the len of output after the q1/q3 cutoffs
	
	return(answer_4) # Should be a list, and must be in the correct order
	
print(question_4())

def question_5():
	df = question_3() 
	df_lsf = df['Land_Square_Feet']

	# Drop rows where there are outliers for Land_Square_Feet from df
	df = df[~df_lsf.isna()] #drop rows from the dataframe where Land Square Feet == NaN

	df['Land_Square_Feet'] = df['Land_Square_Feet'].astype('int64') #convert to int64 from float.  float was returning 'nan' when used with stats.iqr

	df_lsf_int = df['Land_Square_Feet']

	# Calculate the IQR for Land_Square_Feet 
	q5_iqr = stats.iqr(df_lsf_int)

	#Calculate cut-offs for Land_Square_Feet
	q5_iqr_qtr1 = np.percentile(df_lsf_int, 25)
	q5_iqr_qtr3 = np.percentile(df_lsf_int, 75)

	q5_iqr_lower = q5_iqr_qtr1 - (1.5 * q5_iqr)
	q5_iqr_upper = q5_iqr_qtr3 + (1.5 * q5_iqr)

	df = df[(df_lsf_int < q5_iqr_upper) & (df_lsf_int > q5_iqr_lower)] #len of df is: 6998

	# Calculate Pearson's correlation coefficient for df["Land_Square_Feet"] and df["Sale_Price"]
	df.cov() #alternate method for getting covariance
	df_cov = df.cov()
	print(df_cov.iat[12,8])
	
	df_corr = df.corr(method='pearson')
	print(df_corr)
	print(df_corr.iat[12,8])

	#NOTE: the covariance below was my initial attempt at solving the correlation covariance
	#I've since changed my answer from the var: covariance to the var: df_corr (SEE ABOVE)
	covariance_sum = 0
	mean_x = df['Sale_Price'].mean()
	mean_y = df['Land_Square_Feet'].mean()
	n = len(df)
	for x, y in zip(df['Sale_Price'], df['Land_Square_Feet']):
		covariance_sum = (x-mean_x)*(y-mean_y) + covariance_sum
	covariance = covariance_sum/n #Divide by n for the population and n-1 for samples
	
	# Calculate Pearson's correlation coefficient for df["Land_Square_Feet"] and df["Sale_Price"]
	answer_5 = df_corr
	return(answer_5) # Should be a single float
print(question_5())
	

	