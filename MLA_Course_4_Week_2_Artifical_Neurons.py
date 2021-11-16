#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 21:58:12 2021

@author: user
"""

import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

df = pd.read_csv("/Users/user/Downloads/ML Analytics/ML Analytics - Course 4/social_honeypot_sample.csv")
y = np.where(df["Polluter"].values==0, -1, 1)
X = df[["NumberOfFollowings", "NumberOfFollowers", "AgeAtTweet"]].values

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25)

w = np.zeros(X.shape[1])
max_epochs = 3


for epoch in range(max_epochs):
    for x_i, y_i in zip(X_train, y_train):
        z = x_i.dot(w)
        if((y_i*z)<=0):
            w = w + y_i*x_i

#Get our predictions on the training and test sets
zs_train = X_train.dot(w)
y_train_predicted = np.where(zs_train<0, -1, 1)
print('Training accuracy: ', accuracy_score(y_train, y_train_predicted))

zs_test = X_test.dot(w)
y_test_predicted = np.where(zs_test<0, -1, 1)
print('Test Accuracy: ', accuracy_score(y_test, y_test_predicted))

'''
def forward_propagation(inputs, w_hidden, w_hidden_bias, w_output, w_output_bias):
    z_hidden = inputs.dot(w_hidden) + w_hidden_bias
    activated_hidden = sigmoid_activation(z_hidden)
    
    z_output = activated_hidden.dot(w_output) + w_output_bias
    activated_output = sigmoid_activation(z_output)
    
    return(z_hidden, activated_hidden, z_output, activated_output)

def sigmoid_activation(z_value):
    z_activated = 1/(1+np.exp(-z_value))
    return z_activated


def log_loss(y_true, output):
    log_loss = -np.sum((y_true*(np.log(output))+(1-y_true)*np.log(1-output)))
    return log_loss

def predict(inputs, w_hidden, w_hidden_bias, w_output, w_output_bias):
    z_hidden, activated_hidden, z_output, activated_output = forward_propagation(inputs, w_hidden, w_hidden_bias, w_output, w_output_bias)
    predictions = np.where(z_output<0, 0, 1)
    return predictions
'''


   