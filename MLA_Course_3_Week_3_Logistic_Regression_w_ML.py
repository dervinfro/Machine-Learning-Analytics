import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random

from sklearn.metrics import roc_curve # ROC (receiver operating characteristic) - plots false positives against true positives
from sklearn.metrics import roc_auc_score # AUC (Area Under Curve) - a single number that can evaluate the models performace
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/social_honeypot_sample.csv')
X = df[['NumberOfFollowings','NumberOfTweets']]
X_scaled = MinMaxScaler(feature_range=(-5,5)).fit_transform(X) # This the formula that supports the MinMaxScaler: x_scaled = (x-min(x)) / (max(x)–min(x)) 
# https://towardsdatascience.com/preprocessing-with-sklearn-a-complete-and-comprehensive-guide-670cb98fcfb9

y=df['Polluter']

model = LogisticRegression(C=10).fit(X_scaled,y)
print('The mean of accuracy is {}'.format(model.score(X_scaled,y)))

xs, ys = np.mgrid[-5:5:.01, -5:5:.01]
grid = np.c_[xs.ravel(), ys.ravel()]
probs = model.predict_proba(grid)[:, 1].reshape(xs.shape)

f, ax = plt.subplots(figsize=(6, 6))
contour = ax.contourf(xs, ys, probs, 25, cmap="RdBu",
                      vmin=0, vmax=1)
ax_c = f.colorbar(contour)
ax_c.set_label("$P(y = 1)$")
ax_c.set_ticks([0, .25, .5, .75, 1])

ax.scatter(X_scaled[:,0], X_scaled[:, 1], c=y[:], s=50,
           cmap="RdBu", edgecolor="white", linewidth=1)

ax.set(aspect="equal",
       xlim=(-5, 5), ylim=(-5, 5),
       xlabel="$X_1$", ylabel="$X_2$")
#plt.show()


#####################
#####################

def confusion_matrix():
    #NOTE: the predict() returns the predicted labels.  If you want predicted probability, use predict_proba()
    predicted = model.predict(X)
    cnf_matrix = confusion_matrix(y, predicted)  #SEE: Example - Logistic Regression -2.py
    print('CNF MATRIX: ', cnf_matrix, sep='\n')
    
    labels=[0,1]
    fig, ax = plt.subplots()
    tick_marks = np.arange(len(labels))
    plt.xticks(tick_marks, labels)
    plt.yticks(tick_marks, labels)
    sns.heatmap(pd.DataFrame(cnf_matrix), annot=True,fmt="g", cbar=False)
    ax.xaxis.set_label_position("top")
    plt.tight_layout()
    plt.title("Confusion Matrix", y=1.1)
    plt.ylabel("Actual Label")
    plt.xlabel("Predicted label")
    #plt.show()
#confusion_matrix()


#####################
#####################


#NOTE: for the binary classification predict_proba returns the probabilities of being in class 0 or in class 1.
# Here, we are interested in on the probabilities for being class 1
'''
NOTE: Remove the [:,1] and print the output (it will be two columns):
 The first index refers to the probability that the data belong to class 0, and the second refers to the probability that the data belong to class 1. 
 Using [:,1] in the code will give you the probabilities of getting the output as 1. If you replace 1 with 0 in the above code, you will only get the probabilities of getting the output as 0.
 https://datascience.stackexchange.com/questions/22762/understanding-predict-proba-from-multioutputclassifier
 https://discuss.analyticsvidhya.com/t/what-is-the-difference-between-predict-and-predict-proba/67376
'''
def auc():
    predicted_proba = model.predict_proba(X)[:,1] 
    print(predicted_proba)
    
    fpr, tpr, thresholds = roc_curve(y, predicted_proba)
    auc_score = roc_auc_score(y, predicted_proba)
    
    plt.plot(fpr,tpr)
    plt.xlabel('False Pos. Rate')
    plt.ylabel('True Pos. Rate')
    #plt.show()
    
    print('The AUC is {}'.format(auc_score))
auc()


#####################
#####################

X = df['Tweet']

#Read the documentation for all the parameters!
#Here, we set min_df to 10, meaning a word must appear in at least 10 documents to be included
#max_df is set to .7, meaning that if it appears in more than 70% of the documents, we'll exclude it (because it's too common)
#ngram_range is set to (1,2), meaning we'll include individual words and contiguous sequences of 2
bow_transform = CountVectorizer(min_df=10, max_df=.7, ngram_range=(1,2))

#Next, our vectorizer needs to be fit on our dataset. If you've divided into training and test sets, you follow the usual procedure:
#Fit on the training set. Then, transform the training and test sets. DO NOT REFIT ON THE TEST SET.
X_bow = bow_transform.fit_transform(X)

print(X_bow.toarray().shape)

features = bow_transform.get_feature_names()
print(features[:10])

model = LogisticRegression().fit(X_bow,y)
print('The accuracy is: {}'.format(model.score(X_bow,y)))

word_weights = pd.Series(index=bow_transform.get_feature_names(), data=np.abs(model.coef_[0]))
print(word_weights.sort_values(ascending=False)[:5])

#####################
#####################

tfidf_transformer = TfidfTransformer()
X_tfidf = tfidf_transformer.fit_transform(X_bow)

weights = pd.Series(tfidf_transformer.idf_, index=bow_transform.get_feature_names())
top_5_weights = weights.sort_values(ascending=False)[:5]
print(top_5_weights)