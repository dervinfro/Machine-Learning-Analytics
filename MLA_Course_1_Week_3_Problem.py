import statistics
from scipy import stats

value = [24.1,
39.4,
124.2,
56.4,
65.6]
mean = round(sum(value)/len(value), 3)
print('Mean: ', mean)
print('STD: ', round(statistics.stdev(value), 3))
values = [13.20,13.32,13.45,13.66,14.03]
IQR = round(values[3]-values[1], 2)
print('IQR: ', IQR)

#to find outlier and potential outliers
#Q1 - (.15 * IQR) and Q3  ( 1.5 * IQR)
print('Upper Inner Fence: ', values[3] + (1.5 * IQR))



