import scipy.stats as stats
import pandas as pd


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/bathroomsbinned.csv')
df_groupby = df.groupby('Bathrooms').count()
print(df_groupby)
one_bath = df[df['Bathrooms']=='1.0']
two_bath = df[df['Bathrooms']=='2.0']
three_bath = df[df['Bathrooms']=='3.0']
four_bath = df[df['Bathrooms']=='4+']

#the one-way ANOVA test will check whether the mean sale prices for these four groups differ in a way that can be said to be statistically significant.
print(stats.f_oneway(one_bath['Sale_Price'], two_bath['Sale_Price'], three_bath['Sale_Price'], four_bath['Sale_Price']))
#RESULT: F_onewayResult(statistic=791.7087223604174, pvalue=0.0)

