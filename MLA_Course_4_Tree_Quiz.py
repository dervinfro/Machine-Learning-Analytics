#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 12:04:00 2021

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import export_graphviz
from sklearn.model_selection import train_test_split

'''
https://towardsdatascience.com/scikit-learn-decision-trees-explained-803f3812290d
'''


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 4/treequiz.csv')
X = df[["squarefoot", "walkscore", "avg_school_score"]]
df["Log_Value"] = np.log(df["value"])
y = df["Log_Value"]


# Now, we just need to create our test and train sets and use the function above
X_train, X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=25)

model_q2345 = DecisionTreeRegressor(criterion='mse', max_depth=2, random_state=25)
model_q2345.fit(X_train, y_train)


model_q6 = DecisionTreeRegressor(max_depth=10, min_samples_split=5, random_state=25).fit(X_train,y_train)
pruning_path = model_q6.cost_complexity_pruning_path(X_train, y_train)
alphas = pruning_path.ccp_alphas

test_scores = []
train_scores = []

clfs = []
for alpha in alphas:
    clf = DecisionTreeRegressor(ccp_alpha=alpha, random_state=25)
    clfs.append(clf.fit(X_train,y_train))

for clf in clfs:
    train_scores.append(clf.score(X_train, y_train))
    test_scores.append(clf.score(X_test,y_test))

print('Q6 & Q7: ', max(alphas),'//', max(test_scores))


# Defining and fitting a DecisionTreeRegressor instance
model_q10 = AdaBoostRegressor(base_estimator=DecisionTreeRegressor(max_depth=10, min_samples_split=10, random_state=25), n_estimators=100)
model_q10.fit(X_train,y_train)
print('AdaBoost: ', model_q10.score(X_test, y_test))

# Creates dot file named myTreeQuizTrain_Q11.dot
export_graphviz(
            model_q10,
            out_file =  "myTreeQuizTrain_Q10.dot",
            feature_names = list(X.columns),
            filled = True,
            rounded = True)

model_q11 = RandomForestRegressor(n_estimators=100, max_depth=10, min_samples_split=5, random_state=25)
model_q11.fit(X_train, y_train)
print('Random Forest: ', model_q11.score(X_test, y_test))







