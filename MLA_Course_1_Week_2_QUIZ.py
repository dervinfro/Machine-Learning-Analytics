import seaborn as sns
import numpy as np
from scipy.stats import binom
from scipy.stats import norm
from math import factorial 

#question 1
#independent events: both occur -- 𝑃(𝐴∩𝐵)=𝑃(𝐴)∗𝑃(𝐵)
a1 = .2
b1 = .5
prob_both_1 = a1 * b1
print('Question 1: ', prob_both_1)

#question 2
#mutually exclusive: Mutually Exclusive events, the probability of A or B is the sum of the individual probabilities
a2 = .3
b2 = .2
muth_exclu = a2 + b2
print('Question 2: ', muth_exclu)
#REF: https://www.mathsisfun.com/data/probability-events-mutually-exclusive.html

#question 3
#independent events: either occurs -- 𝑃(𝐴∪𝐵)=𝑃(𝐴)+𝑃(𝐵)−𝑃(𝐴∩𝐵)
a3 = .2
b3 = .4
prob_both_3 = a3 * b3
ethr_occur = (a3 + b3) - prob_both_3
print('Question 3: ', ethr_occur)

#question 5
cdf = norm.cdf(20, 12, 4) # X is normally distributed with a mean of 12 and a standard deviation of 4.  What is the prob that x is greater than 20?
print('Question 5: ', round(1- cdf, 4))

#question 6
#top 3 candidates from a candidate pool of 20.  Rank them in preference.
number_candidates = 20
top_candidates = 3
permutation = factorial(number_candidates)/factorial(number_candidates - top_candidates)
print('Question 6: ', permutation)

#question 7
#top 3 candidates from a candidate pool of 20.  Not ranked.
permutation_nr = factorial(number_candidates)/(factorial(20-3) * factorial(3))
print('Question 7: ', permutation_nr)

#question 8
ss = 350 #sample survey
tp = 34958 #total population
pop_75 = 3569 #population over 75
sample_size = ss/tp
print('Question 8: ', sample_size)

#question 9
pp = 150000 #property price
pl = np.log(pp) #price log for value of pp
#print(pl) #value 11.92
print('Question 9: ', round((pl-12.38)/.69, 2)) #log(pp), mean of 12.38 and a standard deviation of .69.  RESULT: z-score is -0.67
#REF: http://www.z-table.com/

#question 10
cdf1 = norm .cdf(pl, 12.38, .69) #log(pp), a mean of 12.38 and a standard deviation of .69.  What's the prob that the property is valued more than $150K
print('Question 10: ', round(1-cdf1, 3)) 

#question 11
pdf = binom.pmf(5,25,.35) #chance of winning an appeal is .35.  25 appeals submitted, what is the prob that 5 will succeed? 
print('Question 11: ', round(pdf, 3))
#question 12
cdf2 = binom.cdf(3,25,.35) #chance of winning an appeal is .35.  25 appeals submitted, what is the prob that 3 or fewer will succeed?
print('Question 12: ', round(cdf2, 3))
#question 13
cdf3 = binom.cdf(8,25,.35) #25 appeals submitted.  What is the prob of more than 8 succeeding?
print('Question 13: ', round(1-cdf3, 3))