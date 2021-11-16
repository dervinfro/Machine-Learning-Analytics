import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
#pd.set_option('display.max_rows', None)
pd.set_option('display.width', 150)
from math import radians, cos, sin, asin, sqrt
# get_distance() takes the latitude and longitude of two locations in degrees and uses the
# Haversine formula to calculate the distance in kilometers

import pandas as pd
#print(df.columns)



def question_1():
	df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment1.csv')
	# Subset to properties coded as "One-Storey Residences"
	property_class_array = [202,203,204] #set array to three property classes
	df = df.loc[df["Property_Class"].isin(property_class_array)]
#	df = df[df['Type_of_Residence'].isin(['1-story'])]
		
	# Drop rows with missing values in Half_Baths and Full_Baths
	df = df[~df['Full_Baths'].isna()] #dropped the full bath NaN
	df = df[~df['Half_Baths'].isna()] #dropped the half bath NaN
#	df['Half_Baths'] = df['Half_Baths']*.5
	answer_1 = df
	
	return (answer_1) # Return the dataframe
#print(question_1())


def question_2():
	df = question_1() # Get the dataframe from question_1() that you've been working on
	# Create column Total_Baths
	Total_Baths = df['Full_Baths'] + df['Half_Baths']*.5
	
	df.insert(9, 'Total_Baths', Total_Baths)
	# Drop columns Full_Baths and Half_Baths
	df = df.drop(df.columns[[10,11]], axis=1)
	# Create the column Log_Sale_Price
	Log_Sale_Price = np.log(df['Sale_Price'])
	df.insert(16, 'Log_Sale_Price', Log_Sale_Price)
	answer_2 = df #when I run df.describe() I get a returned mean value of 1.419629 for Total_Baths.  The filter of the 1-story residence in first function was wrong
	return (answer_2) # Return the dataframe
#print(question_2())


def question_3():
	df = question_2() # Get the dataframe from question_2() that you've been working on
	# Recode the "Basement" column
	#Basement: Basement type - 0 = None, 1 = Full, 2 = Slab, 3 = Partial, 4 = Crawl
	df['Basement'] = df['Basement'].replace(['Full basement', '1.0'], 'Full')
	df['Basement'] = df['Basement'].replace(['crawlspace', 'Slab basement', 'Partial basement', '2.0', '3.0', '4.0'], 'Partial')

	answer_3 = df
	return (answer_3) # Return the dataframe
print(question_3())
	
	
def question_4():
	df = question_3() # Get the dataframe from question_3() that you have been working on
	# Set its index to Tax_Year
	df = df.set_index('Tax_Year')
	# Set the datatype for Central_Air to category
	df['Central_Air'] = df['Central_Air'].astype('category')
	answer_4 = df.sort_values('Tax_Year', ascending=True)
	return (answer_4)
#print(question_4())

print('*' * 25)
print('*' * 5, 'Extra Credit', '*' * 5)
print('*' * 25)



def question_5():
	df = pd.read_csv("/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/assignment1.csv") # Read the properties dataset
	restaurant_data = pd.read_csv("/Users/user/Downloads/ML Analytics/ML Analytics - Course 1/city_data.csv") # Read the dataset you created in the previous part
	
	# Drop rows in restaurant_data that don't have information for lattitude and longitude
	restaurant_data = restaurant_data[~restaurant_data['LATITUDE'].isna()]
	restaurant_data = restaurant_data[~restaurant_data['LONGITUDE'].isna()]
	
	# Subset df to Tax_Year 2018
	df = df[df['Tax_Year'] == 2018]
	
	# Use get_distance to calculate for each property how many bussinesses in dine_in_restaurants are <5 km away
	def get_distance(lat1, lon1, lat2, lon2):
			lat1_in_radians = np.radians(lat1)
			lon1_in_radians = np.radians(lon1)
			lat2_in_radians = np.radians(lat2)
			lon2_in_radians = np.radians(lon2) 
			dlon = lon2_in_radians - lon1_in_radians 
			dlat = lat2_in_radians - lat1_in_radians
			
			a = np.sin(dlat / 2)**2 + np.cos(lat1_in_radians) * np.cos(lat2_in_radians) * np.sin(dlon / 2)**2
			c = 2 * np.arcsin(np.sqrt(a))
			distance = 6372 * c

			return(distance) #returns the distance in km between (lat1, lon1) and (lat2, lon2)
			

	# Add to df that information for each property in a new Rest_Count column
	Rest_Count = get_distance(df['Latitude'], df['Longitude'], restaurant_data['LATITUDE'], restaurant_data['LONGITUDE']).dropna()
	df.insert(17, 'Rest_Count', Rest_Count)
#	answer_5 = restaurant_data[['LATITUDE', 'LONGITUDE']]
	answer_5 = df
	return (answer_5) # Return the new dataframe
	
print(question_5())









	

