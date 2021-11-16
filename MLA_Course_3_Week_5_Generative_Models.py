import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

df = pd.read_csv("/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/social_honeypot_sample.csv")
X = df['Tweet']
y = df['Polluter']


bow_transform = CountVectorizer(min_df=20, max_df=.7, ngram_range=(1,3))
X_bow = bow_transform.fit_transform(X)

from sklearn.model_selection import cross_val_score

from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

nb_clf = GaussianNB()
lda_clf = LinearDiscriminantAnalysis()
qda_clf = QuadraticDiscriminantAnalysis()

#Note that we need to use .toarray() with these methods; they won't take a sparse matrix
nb_cv = cross_val_score(nb_clf, X_bow.toarray(), y, cv=2)
lda_cv = cross_val_score(lda_clf, X_bow.toarray(), y, cv=2)

#QDA is particularly computationally intensive and may take a few minutes
qda_cv = cross_val_score(qda_clf, X_bow.toarray(), y, cv=2)

print("For Naive Bayes, the CV score is %f" % nb_cv.mean())
print("For LDA, the CV score is %f" % lda_cv.mean())
print("For QDA, the CV score is %f" % qda_cv.mean())