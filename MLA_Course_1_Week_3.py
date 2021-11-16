import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
import seaborn as sns
import statsmodels.api as sm
import statsmodels.api as sm


from scipy import stats
from scipy.stats import skew
from scipy.stats import kurtosis


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/module2examplesv2.csv')
df_sp = df['Sale_Price']
'''


df_sp_mean = df_sp.mean() #dataframe Sale_Price mean
df_sp_median = df_sp.median() #dataframe Sale_Price median
print('Mean: ', df_sp_mean, '\nMedian: ', df_sp_median)

if df_sp_mean > df_sp_median:  
	print('Right-skewed: Mean is GREATER than Median') #right skewed: mean > median
else:
	print('Left-skewed: Mean is LESS than Median') #left skewed: mean < median
	
ax = sns.distplot(df['Sale_Price'])
ax.set_xlabel = 'Sale_Price'
plt.show()

#Trimmed Mean
sp_trim_mean = stats.trim_mean(df_sp, .05) #trimming .05 off the top and bottom of the distribution
print('Trimmed Mean: ', sp_trim_mean)

#Range
print('Range: ', df_sp.max() - df_sp.min())

#sample standard deviation
print('Sample Standard Deviation: ', df_sp.std())

#quartiles
print('Quantile: ', df_sp.quantile([.25,.5,.75]))


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/batterycountvstemp.csv')
sns.scatterplot(y=df['Battery_Count'], x=df['Temperature'])


df1 = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/scatterexamples.csv')
plt.subplot(2,2,1)
sns.scatterplot(x=df1['x'], y=df1['y']).set_title('Negative Corr.')
plt.subplot(2,2,2)
sns.scatterplot(x=df1['x'], y=df1['y2']).set_title('Positive Corr.')
plt.tight_layout()
plt.show()

#Crosstabs
df_vd = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/targetselection.csv') #visualize data
print(pd.crosstab(df_vd['Target'], df_vd['Agent'], margins=True))


#Histograms -- module2examplesv2.csv continued from above
plt.figure(figsize= (8,5))
ax = sns.distplot(df_sp, bins=100, kde=False)
ax.set_xlabel('Sale Price')
ax.set_ylabel('Frequency')
plt.show()
'''

#NOTE: Skewness tells us about histogram symmetry
#Kurtosis is about the thickness of those tails
#Skew = data symmetry; whether the data falls equally around the mean or whether a tail goes left or right.
# ..for normally distributed data, the skewness should be about zero.
distributsions = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/skeweddistributions.csv')
low_skew = distributsions['Normal_Data'] # mean/median/mode roughly align in normally distributed data.
pos_skew = distributsions['Pos_Skew_Data'] # more data is to the right of the peak(right-skewed).  The mean is more that the median.
neg_skew = distributsions['Neg_Skew_Data'] # more data is the left of the peak(left-skewed).  The mean is less than the median.
print(low_skew, pos_skew, neg_skew)
plt.figure(figsize=(10,4))
plt.subplot(1,3,1) # 1 row // 3 graphs //1st position
sns.displot(low_skew, kde=False)
plt.title('Low Skew')

plt.subplot(1,3,2)# 1 row // 3 graphs // 2nd position
sns.displot(pos_skew, kde=False)
plt.title('Pos Skew Data')

plt.subplot(1,3,3) # 1 row, // 3 graphs // 3rd position
sns.displot(neg_skew)
plt.title('Neg Skew Data')
plt.show()



df_sp_skew=skew(df_sp, bias=False) #dataframe sale price skew // if bias set to False, then calculation are corrected for bias.
print('DF SP Skew: ', df_sp_skew)
'''
#Kurtosis = a related measure that helps describe the shape of the distributions.  Kurtosis is the thickness of the tails. 
#The bigger the tails; the bigger the kurtosis.  
#The bigger the kurtosis; the more data outliers.
k = np.linspace(-5, 5,100)
normal_dist = scipy.stats.norm(loc=0, scale=1)
t_dist = scipy.stats.t(df=1)
binom_dist = scipy.stats.binom(n=50, p=.5)

plt.figure(figsize=(15,5))
plt.subplot(1,3,1)
plt.plot(k, normal_dist.pdf(k))
plt.ylim(0,.4)
plt.title('Mesokurtosis')

plt.subplot(1,3,3)
plt.plot(k, t_dist.pdf(k))
plt.ylim(0,.4)
plt.title('Leptokurtosis')

plt.subplot(1,3,2)
sns.distplot(k, binom_dist.pmf(k))
plt.ylim(0,.4)
plt.title('Platykurtosis')
plt.show()
'''
kurt_df_sp = kurtosis(df_sp, bias=False, fisher=False) #kurtosis dataframe sale price
print('Kurtosis: ', kurt_df_sp) #return value of: 51.97770537158306. High Kurtosis value indicates many potential outliers

#Boxplots: They depict min, max, median, first quartile, third quartile.
sns.boxplot(df_sp) #boxplot dataframe sale price
plt.show()
'''
#Quantile - Quantile Plots
cleaned_data = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/salenooutliers.csv')
sm.qqplot(cleaned_data['Sale_Price'], line="45", dist=scipy.stats.distributions.norm)
plt.show()
'''