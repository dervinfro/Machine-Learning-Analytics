import pandas as pd
import numpy as np
import timeit

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score 
from sklearn.metrics import mean_squared_error 
from sklearn.preprocessing import MinMaxScaler 
from sklearn.pipeline import make_pipeline 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

pd.options.display.max_rows=None
pd.options.display.max_columns=None
pd.options.display.width= 175


df = pd.read_csv('/Users/user/Downloads/ML Analytics/ML Analytics - Course 3/social_honeypot_sample.csv')

y = df['Polluter']
X = df[['LengthOfDescriptionInUserProfile','LengthOfScreenName','NumberOfFollowings','NumberOfFollowers','NumberOfTweets']] # used for Question 7
#X = df['Tweet'] # used for Question 8,9,10,11

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=25)


def Question_7():
    '''
    Create a logistic regression using the above features and default paramters (l2 penalty and C=1). What's the mean precision score from 10-fold cross-validation? Round to three decimal places.
    '''
    
    model = LogisticRegression(penalty='l2',C=1)
    
    val_scores = cross_val_score(model, X_train, y_train, scoring='precision', cv=10)
   
    print('Q7:', round(np.mean(val_scores),3)) # Correct: 0.889
Question_7()



def Question_8():
    '''
    First, you'll take a simple bag of words approach. Create a logistic regression the training set with CountVectorizer(). Use default parameters. Conduct 10-fold cross-validation, using precision for scoring.

What mean precision does this model get? Round to three decimal places.
    '''
    vectorizer = CountVectorizer()
    X_new = vectorizer.fit_transform(X_train)    
    
    model = LogisticRegression()
    
    val_scores = cross_val_score(model, X_new, y_train, scoring='precision', cv=10)
    print('Q8: ', round(np.mean(val_scores),3)) # Correct: 0.749
    
    
Question_8()

def question_9():
    '''
    Next, create a logistic model where you preprocess the data with TfidfVectorizer and then evaluate it with 10-fold cross validation. Recall that you may get an overly-optimistic evaluation if you use use the Tfidfvectorizer on all of X_train, and then conduct cross-validation on just the logistic regression. Instead, during cross-validation, you'll want to fit the vectorizer on just the trainng data, fit the regression on the training data, and then test it on the hold-out set.

    To do this, you'll need to create a pipeline with the vectorizer and the logistic regression, and conduct the cross-validation using the pipeline. To start with, again, use default parameters. What mean precision do you get? Round to three decimal places.
    '''
    
    tfidf_vectorizor = TfidfVectorizer()
    X_vector = tfidf_vectorizor.fit_transform(X_train)
    
    model = LogisticRegression().fit(X_vector, y_train) 
    
    pipeline = make_pipeline(tfidf_vectorizor, model) #NOTE: My pipeline attributes were initially incorrect.  I was passing the X_vector AND NOT the tfidf_vectorizor.  Now the vectorizor is being passed.
    # https://stackoverflow.com/questions/58543937/how-to-save-tfidf-vectorizer-in-scikit-learn
    
    val_score = cross_val_score(pipeline, X_train, y_train, scoring='precision', cv=10)
    print('Q9: ', round(np.mean(val_score),3)) #Correct: 0.747
    
    
question_9()

def question_10():
    '''
    Conduct a Grid Search to find the best values for these parameters. Use precision as the scoring method, but this time only do 5-fold cross-validation to help with the computation time.

    If you need to understand how the parameters in the parameter grid for the Grid Search need to be named, you can use my_grid_search.estimator.get_params().keys() for a list of the names of the parameters.

    What's the best score from the Grid Search? Round to three decimal places.
    '''
    start_time = timeit.default_timer()
        
    tfidf_vectorizor = TfidfVectorizer()
    
    model = LogisticRegression() # The model DOES NOT need to be fit PRIOR being fed into the pipeline.  Fit before the pipeline only slows down the code. 
    
    #parameters = {'tfidfvectorizer__max_df':(0.7,0.8,0.9,1.0),
                  #'tfidfvectorizer__min_df':(1,5,10),
                  #'tfidfvectorizer__ngram_range':[(1,1),(1,2)],
                  #'tfidfvectorizer__use_idf':['True','False']}
                  
    parameters = {'tfidfvectorizer__max_df':[0.7],
                  'tfidfvectorizer__min_df':[5],
                  'tfidfvectorizer__ngram_range':[(1,2)],
                  'tfidfvectorizer__use_idf':['True']}    
    
    pipeline = make_pipeline(tfidf_vectorizor, model) #NOTE: My pipeline attributes were initially incorrect.  I was passing the X_vector AND NOT the tfidf_vectorizor.  Now the vectorizor is being passed.
    # https://stackoverflow.com/questions/58543937/how-to-save-tfidf-vectorizer-in-scikit-learn  
    #print(pipeline.get_params().keys())
    
    grid_search = GridSearchCV(estimator=pipeline, param_grid=parameters , scoring='precision', cv=5)
    grid_search.fit(X_train, y_train)
    X_tweet = pd.Series(['Everything is getting better. http://bit.ly/6DcsUR'])
    print('Q10: ', round(grid_search.best_score_,3))
    print('Q11: ', grid_search.best_estimator_.steps)
    print('Q12: ', grid_search.predict_proba(X_tweet))
    
    print('Total time: ', timeit.default_timer() - start_time)
    #Correct: Q10:  0.753
    #Correct: Q11:  [('tfidfvectorizer', TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=(1, 2), use_idf='True')), ('logisticregression', LogisticRegression())]
    #Correct: Q12:  [[0.42529456 0.57470544]]
    
question_10()
'''
Q8:  0.749
Q9:  0.747
Q10:  0.753
Q11:  [('tfidfvectorizer', TfidfVectorizer(max_df=0.7, min_df=5, ngram_range=(1, 2), use_idf='True')), ('logisticregression', LogisticRegression())]
Q12:  [[0.42529456 0.57470544]]
Total time:  1.4935354730000006
'''



