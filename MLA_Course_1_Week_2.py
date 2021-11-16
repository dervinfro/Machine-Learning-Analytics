import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import random
from numpy.random import randint
from scipy.stats import norm
from scipy.stats import binom


n = 500 #number of trials
size = 1000 # number of iterations to run the trial

successes = []
for k in range(size):
	success = 0
	for k in range(n):
		flip = random.randint(0,1)
		if flip == 1:
			success = success+1
	successes.append(success)
ax = sns.distplot(successes, hist=False, kde=True) #try this w/o hist and with kde=False
ax.set_xlabel("Heads", fontsize=16)
ax.set_ylabel("Tails", fontsize=16)
kde_x, kde_y = ax.lines[0].get_data()
ax.fill_between(kde_x, kde_y, where=(kde_x < 250), color='r')
plt.show()

print('*' * 23)
print('*' * 5, 'SCIPY STATS', '*' * 5)	
print('*' * 23)


pmf = binom.pmf(250, 500, 0.5) #calculate the pmf of getting 250 heads out of 500 flips, where p = .5
cmf = binom.cdf(250, 500, 0.5) #calculate the cmf for the same
print("For this sample of {} flips in {} experiments, the pmf is {} and the cmf is {}".
		format(n, size, round(pmf, 4), round(cmf, 4)))
		
print()
cdf = norm.cdf(30, 24.5, 4.7)
pdf = norm.pdf(30, 24.5, 4.7)
print('The probability of a BMI of 30 is {}'.format(round(pdf, 4)))
print('The probability of a BMI of 30 or less is {}'.format(round(cdf, 4)))
print('The probability of a BMI of 30 or more is {}'.format(round(1-cdf, 4)))


print()
df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment1.csv')
sns.distplot(df['Sale_Price'])
plt.show()

print()
df = df[df['Sale_Price'] > 150000]
ax = sns.distplot(np.log(df['Sale_Price']))
ax.set_xlabel('Log(Sale_Price)')
plt.show()


print('*' * 28)
print('*' * 2, 'Central Limit Theorem', '*' * 2)
print('*' * 28)

#Below: CLT with Normally Distributed Population
#Example: 20 sided dice

n = 10 #number of rolls 
k = 50 #number of samples

sample_means = []
for x in range(0,k):
	roll_mean = np.mean(randint(1,21,n))
	sample_means.append(roll_mean)
	
ax = sns.distplot(sample_means, kde=False)
ax.set_xlabel('Means of {} rolls'.format(n))
ax.set_ylabel('Frequency')
plt.show()
print('The mean of means is: {}'.format(np.mean(sample_means)))
'''

#Below: CLT without Normally Distributed Population
np.random.seed(1427)
pop = np.random.poisson(1,100000)
ax = sns.distplot(pop, kde=False)
plt.show()
'''