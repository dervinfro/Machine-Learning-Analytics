import statistics
import pandas
import numpy

np_ran = numpy.random.randint(0,50,50)
stat_ran = statistics.stdev(np_ran)
stat_mean = statistics.mean(np_ran)
print('Stdev: ', stat_ran)
print('Mean: ', stat_mean)
for x in range(1,4):
	print('Stdev {} is {}'.format(x, x*stat_ran)) # 144 // 288 // 432
	
#Q7 
row = 1630
column = 825
total = 3433
print(round((row*column)/total)) #391.7127876492863

#Q8
Full_basement_m=365777; #n=427

Partial_basement_m=281903; #n=275

No_basement_m=236977; #n=298

Total_mean = 304329 #(n=1000)

mean = [365777, 281903, 236977]
number = [427, 275, 298]

list = []
for x in range(0,3):
	result = (number[x]*(mean[x]-Total_mean)**2)
	list.append(result)

print(sum(list)/2) #1551205152950.0

