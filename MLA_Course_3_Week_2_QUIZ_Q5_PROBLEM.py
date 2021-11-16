#!/usr/bin/env python3

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor # fifth question
from sklearn.pipeline import make_pipeline # fifth question
from sklearn.preprocessing import StandardScaler # fifth question
from sklearn.model_selection import cross_val_score # fifth question

pd.options.display.max_columns=None
pd.options.display.max_rows=None
pd.options.display.width=175
#pd.options.display.float_format='{:.3f}'.format

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/quiz2data.csv')	


	
df['log_value'] = np.log(df['value'])

df['const'] = 1

X = df[['const','medhinc','elem_score','walkscore']]
y = df['log_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25) 

pipeline = make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=10, weights='distance', p=1))
print(pipeline)


val_score = cross_val_score(pipeline, X_train, y_train, cv=5)
print(val_score)
print('Val Score Mean: ', round(val_score.mean(),3))
