#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 21:30:04 2021

@author: Derek Frost
"""
'''
MLP for the MNIST dataset
A few key components to refine the accuracy score:
    epochs
    optimizer
    Dense unit values
    Dropout rate
'''
import pandas as pd
import matplotlib.pyplot as plt

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten

df_train = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 4/MNIST_train_sample.csv')
df_test = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 4/MNIST_test_samplev2.csv')

y_df_train = df_train['label']
X_df_train = df_train.drop('label', axis=1) #this drops the 'label' column

y_df_test = df_test['label']
X_df_test = df_test.drop('label', axis=1) #this drops the 'label' column

# the shape of X_df_train is: (8000, 784)....this is for my own sanity.
# the output of following reshape will be: (8000, 28, 28)....again, sanity check.
# the reason for the reshape is that it is required for the following 'for' loop
X_df_train_2dim =  X_df_train.values.reshape(-1,28,28)

'''
Create a subplot of the first six MNIST images.  The subplot is made up of 2 rows and 3 columns.  
This for loop of the subplot is just for me to confirm what's being sent to the model.
'''

# create a loop that displays the first six (6) images
fig = plt.figure(figsize=(10,10))
for i in range(6):
    ax = fig.add_subplot(2, 3, i+1, xticks=[], yticks=[])
    ax.imshow(X_df_train_2dim[i])
    ax.set_title(str(y_df_train.iloc[i]))
    
'''
The following list print lines are just to ensure/validate what values are being output from the 
respective dataframes of: y_df_train and y_df_test.
'''
# print the integer values of the y_df_train dataframe values.....ie [9, 1, 6, 2, 1, 3]
print('Integer value labels ( Y Train ):')
print(list(y_df_train[:6]))
print('')
# print the integer values of the y_df_train dataframe values.....ie [9, 5, 8, 8, 9, 4]
print('Integer value labels ( Y Test ):')
print(list(y_df_test[:6]))
print('')

'''
Given the 'Y' values range from 0-9, this utility will create an array with 10 binary values
 and the respective integer value will be flipped to 1.  All else values will be 0.
Example: the first value of the list (see previous command) is 9. 
    The following binary values correspond to that value: 0. 0. 0. 0. 0. 0. 0. 0. 0. 1.
'''
# one-hot encode the labels.  
y_train = np_utils.to_categorical(y_df_train, 10)
y_test = np_utils.to_categorical(y_df_test, 10)

#print one hot labels
print('One Hot Labels ( Y Train ):')
print(y_train[:10])
print('')
print('One Hot Labels ( Y Test ):')
print(y_test[:10])

#define the model
model = Sequential()
model.add(Flatten(input_shape=X_df_train.shape[1:])) # this value is already flat.  I just wanted to show that Flattening is a key part of MLP.
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

#summarize the model
print(model.summary())

#compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

#fit the model
model.fit(X_df_train, y_train, epochs=4, batch_size=128, verbose=1, validation_split=0.2)

#score the model using the test dataset
score = model.evaluate(X_df_test, y_test, verbose=1)

print('The test accuracy is {}:'.format(100*score[1])) # accuracy is 93.15%




