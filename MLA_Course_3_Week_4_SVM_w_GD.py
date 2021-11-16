import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from numpy.random import random
from sklearn.svm import SVC

def get_hinge_loss(X, w, y, lamb):
    N = len(y)
    losses = 1-y.ravel()*np.dot(X,w)
    losses[losses<0] = 0
    hinge_loss = np.sum(losses)/N
    regularizer = (lamb/2)*np.dot(w,w)
    loss = regularizer + hinge_loss
    return loss

def get_gradient(X, w, y, lamb):
    margins = y.ravel()*np.dot(X,w)
    
    gradients = []
    
    for margin, X_i, y_i in zip(margins, X, y):
        if margin<1:
            gradient = lamb*w - y_i*X_i
        else:
            gradient = lamb*w
        gradients.append(gradient)
        
    average_gradient = sum(gradients)/len(y)
    return average_gradient


def SGD_SVM(X, y, lr, batch_size, max_epochs, lamb):
    w = np.zeros(X.shape[1])
    losses = []
    target = .001
    old_loss = 1
    count = 0
    while (count<max_epochs):
        shuffled_index = np.random.permutation(X.shape[0])
        batch_starts = range(0, X.shape[0], batch_size)
        
        for start_index in batch_starts:
            batch = shuffled_index[start_index:start_index + batch_size]
            x_batch = X[batch]
            y_batch = y[batch]
            gradient = get_gradient(x_batch, w, y_batch, lamb)
            w = w-lr*gradient
            
        current_loss = get_hinge_loss(X, w, y, lamb)
        losses.append(current_loss)
        gain = (old_loss - current_loss)/np.abs(old_loss)
        old_loss = current_loss
        
        
        if (gain<target):
            lr = lr/2
        count = count +1
    
    return (w, losses)


X = random((1000,2)) * 2 -1
y = 2 * (X.sum(axis=1) > 0 ) - 1.0

lr = .01
batch_size = 32
max_epochs = 500
lamb = 1/10000 #low lambda means a low regularization
w, losses = SGD_SVM(X, y, lr, batch_size, max_epochs, lamb)

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('loss')
plt.show()


x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100),indexing='ij')
data = np.stack([xx, yy], axis=2).reshape(-1, 2)
pred = np.dot(data,w).reshape(xx.shape)
plt.contourf(xx, yy, pred,levels=[-0.001, 0.001],extend='both',alpha=0.8, cmap=cm.get_cmap("Spectral"))
flatten = lambda m: np.array(m).reshape(-1,)
plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]),c=flatten(y), cmap=cm.get_cmap("Spectral"))
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.plot()

np.random.seed(1)
X = random((150, 2)) * 2 - 1
y = 2 * ((X ** 2).sum(axis=1) - 0.5 > 0)-1

def plot_svm_boundary(X, y, svm, grid_size=100):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid_size), np.linspace(y_min, y_max, grid_size), indexing='ij')
    data = np.stack([xx, yy], axis=2).reshape(-1, 2)
    pred = svm.predict(data).reshape(xx.shape)
    plt.contourf(xx, yy, pred, cmap=cm.get_cmap("Spectral"), levels=[-0.001, 0.001], extend='both', alpha=0.8)
    flatten = lambda m: np.array(m).reshape(-1,)
    plt.scatter(flatten(X[:, 0]), flatten(X[:, 1]), c=flatten(y), cmap=cm.get_cmap("Spectral"))
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.plot()
    plt.show()
    
C = 1 # used for regularization. Corresponds to the inverse of lambda
gamma = 1 # used to set the radius of influence
svc_clf = SVC(kernel='rbf', random_state=0, gamma=gamma, C=C)
svc_clf.fit(X,y)
plot_svm_boundary(X, y, svc_clf)
