# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 11:47:38 2016

@author: Yogesh
"""

# importing packages
import pandas as pd
import glob
import codecs
from nltk.corpus import stopwords
from stemming.porter2 import stem
import random
from sklearn.naive_bayes import MultinomialNB
from textblob.classifiers import NaiveBayesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
import scipy.sparse as sp
from collections import Counter
import datetime
import string
import re
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn import svm, datasets

from nltk.corpus import twitter_samples

#Importing Dataset of Twitter or facebook
twitter_data = pd.read_csv("Twitter_reach.csv", encoding="ISO-8859-1")
facebook_data = pd.read_csv("Facebook_reaach.csv",encoding="ISO-8859-1")
Tweets = twitter_data.Post
Category= twitter_data.Category
Tweets_list = list(Tweets)
Category_list = list(Category)
Tweets_list[:10]
Category_list = list(Category)
Category_list[:10]

#Partitioning Dataset  into training 
Tweets_train = Tweets_list[0:1000]
Category_train = Category_list[0:1000]
processed_train = []
Tweets_test = Tweets_list[1000:]
Category_test = Category_list[1000:]
 
# Procedure for preprocessing text data
len(Tweets_train)
def preprocessText (text, rmpunc='y', rmdigits='y', removelink='y',  lowerwords='y', minwordlen=1, rmstopwords='y', stemwords='y', uniquewords='n'):
        
    # remove punctuation
    if (rmpunc=='y'):
        regex = re.compile('[%s]' % re.escape(string.punctuation))
        text = regex.sub('', text)
        
    # remove digits
    if(rmdigits=='y'):
        regex = re.compile('[%s]' % re.escape(string.digits))
        text = regex.sub('', text)
        
    if (removelink =='y'): 
        text = re.sub(r"http\S+", "", text)
        #text = re.sub(r"?","",text)
        

    # lower case words
    if (lowerwords=='y'):
        text = text.lower()
        
    # remove short words
    if (minwordlen>1):
        text = ' '.join([word for word in text.split() if (len(word)>=minwordlen)])
        
    # remove stop words
    if (rmstopwords=='y'):
        text = ' '.join([word for word in text.split() if word not in stopwords.words("english")])
        
    # stem the words
    if (stemwords=='y'):
        text = ' '.join([stem(word) for word in text.split()])
            
    # retain only unique words
    if (uniquewords=='y'):
        text = ' '.join(set(text.split()))
        
    # return preprocessed data
    return text

# Applying Pre Processing to training and test set
processed_train = [preprocessText(tweet) for tweet in Tweets_train]
processed_train = [preprocessText(tweet, stemwords ='n', rmpunc = 'n') for tweet in Tweets_train
#processed_train = [preprocessText(tweet, stemwords ='n', rmpunc = 'n', lowerwords ='n') for tweet in Tweets_train]



# Create TFIDF vecotrizer
vectorizer = TfidfVectorizer(encoding='latin1', ngram_range=(2,2), lowercase=FALSE)
X_train = vectorizer.fit_transform(processed_train)
# Create SVD classifier for feature reduction
SVDed = TruncatedSVD()
svded = TruncatedSVD(n_components=500, random_state=42)
transformed_train = svded.fit_transform(X_train)
svded.explain

processed_test = [preprocessText(tweet, stemwords ='n', rmpunc = 'n', lowerwords ='n') for tweet in Tweets_test]
X_test = vectorizer.transform(tweet for tweet in processed_test)
transformed_test = svded.transform(X_test)


X_train = vectorizer.fit_transform(tweet for tweet in processed_train)
assert sp.issparse(X_train)

# Intiaing different classifier algorithm
y_train = Category_train
Category= twitter_data.Category
Category_list = list(Category)
Category_train = Category_list[0:1000]
clf= MultinomialNB()
clf1 = SGDClassifier()
clf2= MultinomialNB()
clf3 = SGDClassifier()
clf4= RandomForestClassifier(n_estimators = 100, max_depth = 300)
clf5 = RandomForestClassifier(n_estimators = 50, max_depth = 100)
clf6 = svm.SVC()

# Fitting different algorithm to the training set
clf.fit(X_train, y_train)
clf1.fit(X_train,y_train)
clf3.fit(transformed_train,y_train)
clf4.fit(X_train,y_train)
clf5.fit(transformed_train,y_train)
clf6.fit(X_train,y_train)
clf5.fit(transformed_train,y_train)

# Testing the algorithm against the test set
pred1 = (clf.predict(X_test))
pred2 = (clf1.predict(X_test))
pred3 = (clf3.predict(transformed_test))
pred4= clf4.predict(X_test)
pred5 = clf5.predict(transformed_test)
pred6= clf6.predict(X_test)

# Building the confusion matrix
cm1 = confusion_matrix(y_test, pred1) # Hit rate 72%
cm2 = confusion_matrix(y_test, pred2)# Hit rate 6%
cm3 = confusion_matrix(y_test, pred3)# Hit rate 75%
cm4= confusion_matrix(y_test,pred4)# Hit rate 68%
cm5 = confusion_matrix(y_test,pred5)# Hit rate 71%
cm6 =confusion_matrix(y_test,pred6)# Hit rate 67%

