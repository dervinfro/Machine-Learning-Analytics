#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  2 15:11:24 2021

@author: user
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 4/noisy_fashion_mnist.csv')

y = df['label']
X = df.drop('label', axis=1)

image = np.array(X.sample(1)).reshape(28,28)
plt.imshow(image, cmap='gray')

