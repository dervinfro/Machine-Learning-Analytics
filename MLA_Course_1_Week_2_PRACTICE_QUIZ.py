from scipy.stats import binom
from scipy.stats import norm
from math import factorial
import numpy as np

print('*' * 28)
print('*' * 3, 'Prob Mass Func (PMF)', '*' * 3)
print('*' * 3, 'Cum. Dist. Func (CMF)', '*' * 3)
print('*' * 28)



pmf1 = binom.pmf(4,4,0.7) # 4 heads // 4 flips /// Heads Probability = 0.7 (Question 1) 
print(round(pmf1, 4))

pmf2 = binom.cdf(3,6,0.7) # 3 heads or less // 6 flips // heads probability of 0.7 -- 0.0595 (Question 2)
print(round(pmf2,4))

print(round(1-pmf2, 4)) # 3 heads or more // 6 flips // heads probability of 0.7 (Question 3)

print(0.1852 - 0.0595) #difference between 2 out of 6 heads (0.0595) and the probability that the next flip will be a head. (3 out of 6 flips is 0.1852) Question 4

n = 10 #number of webpages
k = 3 #size of the number of groups

permutations = factorial(n)/factorial(n-k) #10 webpages all hyperlinked to each other.  How many paths to take to travel between any 3. (Question 7)
print(permutations)

combinations = factorial(n)/(factorial(n-k) * factorial(k))
print(combinations) #how many three page groups are there for ten webpages?

cdf = norm.cdf(3,8,2) #normal distribution with mean of 8; standard deviation of 2.  What is the prob that the variable is less than 3?
print(round(cdf, 3))

cdf1 = norm.cdf(11,8,2) #normal distribution with a mean of 8; standard deviation of 2. What is the prob that the variable is more than 11?
print(round(1-cdf1, 3))

print((11-8)/2) #z score for value of 11; mean of 8 and standard deviation of 2





