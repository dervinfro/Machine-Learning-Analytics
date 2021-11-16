import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

X = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/MNIST_X_sample.csv')
y = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/MNIST_y_sample.csv')


def show_MNIST_image(image):
    fig = plt.figure()
    image = image[-28**2:]
    image = image.reshape(28,28)
    ax = fig.add_subplot(1,1,1)
    plt.axis('off')
    imgplot = ax.imshow(image, cmap=mpl.cm.Greys)
    imgplot.set_interpolation('nearest')
    ax.xaxis.set_ticks_position('top')
    ax.yaxis.set_ticks_position('left')
    plt.tight_layout()
    plt.show()

#show_MNIST_image(X.iloc[55].values)
#print(y.iloc[55])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

def softmax(scores):
    scaled_scores = scores-scores.max(axis=1, keepdims=True) #subtract the maximum to prevent overflow
    exp_scores = np.exp(scaled_scores)
    softmax_scores = exp_scores/(np.sum(exp_scores,axis=1, keepdims=True))
    return softmax_scores

def get_log_loss(X,w,y):
    scores = np.dot(X,w)
    predictions = softmax(scores)
    log_likelihood = np.sum(y*np.log(predictions))/len(y)
    log_loss = -log_likelihood
    return log_loss

def get_gradient(X,w,y):
    scores = np.dot(X,w)
    predictions = softmax(scores)
    error = y - predictions
    
    gradient = -(np.dot(X.T, error))/len(y)
    return gradient

def SGD_softmax(X, y, lr, batch_size, max_epochs):
    w = np.zeros([X.shape[1], y.shape[1]])
    old_loss = 1000
    losses = []
    target = .01
    
    count = 0
    while (count < max_epochs):
        shuffled_index = np.random.permutation(X.shape[0])
        batch_starts = range(0, X.shape[0], batch_size)
        
        for start_index in batch_starts:
            batch = shuffled_index[start_index:start_index + batch_size]
            x_batch = X[batch]
            y_batch = y[batch]
            gradient = get_gradient(x_batch, w, y_batch)
            w = w - lr * gradient
            
        current_loss = get_log_loss(X, w, y)
        gain = (old_loss - current_loss)/np.abs(old_loss)
        losses.append(current_loss)
        old_loss = current_loss
        
        if (gain<target):
            lr = lr/2
            
            
        count = count+1
        
    return (w, losses)

max_epochs = 15
batch_size = 32
lr=.01
X_scaled = StandardScaler().fit_transform(X_train)

const = np.ones((X_scaled.shape[0],1))
X_biased = np.concatenate([const,X_scaled],1)

w, losses = SGD_softmax(X_biased, y_train.values, lr, batch_size, max_epochs)

plt.plot(losses)
plt.xlabel('Iteration')
plt.ylabel('Loss')
#plt.show()

predictions = np.argmax(np.dot(X_biased, w), axis=1)
actual = np.argmax(y_train.values, axis=1)
right = 0 
i = 0
while (i<len(y_train)):
    if predictions[i]==actual[i]:
        right = right + 1
    i = i + 1
print('total correct: ', right)
print('Accuracy: ', right/len(y_train))

#labels = y.columns
#i = 0
#fig = plt.figure(figsize=(20,10))

#for label in labels:
    #plt.subplot(2,5,i+1)
    #plt.imshow(w[-28**2:, i].reshape(28,28))
    #i = i+1
#plt.show()
