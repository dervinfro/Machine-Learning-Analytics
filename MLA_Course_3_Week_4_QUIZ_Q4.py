#!/usr/bin/env python3
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC # Question 4
from sklearn.multiclass import OneVsRestClassifier # Question 4
from sklearn.pipeline import make_pipeline # Question 4
from sklearn.svm import SVC # question 5
from sklearn.model_selection import GridSearchCV # Question 5
from sklearn.metrics import accuracy_score # from Tim

'''
Sandals are label 5.

Sneakers are label 7.

Ankle boots are label 9
'''

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/fashion_MNIST_shoes.csv')

dummies = pd.get_dummies(df['label'])

X = df.drop('label', axis=1)
y = dummies


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

model = make_pipeline(StandardScaler(), OneVsRestClassifier(LinearSVC(C=1, random_state=1)))
model.fit(X_train, y_train)
print('Q4: ', model.score(X_test,y_test)) # result is: Q4:  0.8333333333333334 ** CORRECT **


yhat = model.predict(X_test)
print('Accuracy:', accuracy_score(y_test, yhat))