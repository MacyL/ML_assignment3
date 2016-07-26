#!/usr/bin/env python3
######################## All The library I Need ################################
import pandas as pd
import scipy as sp
import numpy as np
import theano
import pickle
import keras.preprocessing.text
from optparse import OptionParser
import sys, os, re, csv, glob
from time import time
from collections import Counter
from nltk.tokenize import RegexpTokenizer
from nltk.util import ngrams
from sklearn import cross_validation
from sklearn.feature_extraction.text import TfidfVectorizer,HashingVectorizer, TfidfTransformer, CountVectorizer
from sklearn import metrics
from numpy import array
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sknn.mlp import Classifier, Layer
from keras.preprocessing.text import Tokenizer
from keras.layers import Input, Dense,Dropout, Activation, Flatten,Convolution1D, MaxPooling1D
from keras.models import Sequential,Model
#############################################################################################
#Preprocessing
#############################################################################################
# read in all the files to put in a large python list.
pathPos = "/home/admin1/Documents/Machine_learning/assignment3/pos/*.txt"
pathNeg = "/home/admin1/Documents/Machine_learning/assignment3/neg/*.txt"
filesPos = glob.glob(pathPos)
filesNeg = glob.glob(pathNeg)   
FullData=pd.DataFrame()
for name in filesPos:
	f=open(name,'r').read()
	f=re.sub(r'\d+','',f)
	# Remove duplicated white space.
	f=" ".join(f.split())
	isPos=["Positive"]
	FullData=FullData.append(pd.DataFrame({'isPos':isPos, 'text':f}))

for names in filesNeg:
	f=open(names,'r').read()
	f=re.sub(r'\d+','',f)
	# Remove duplicated white space.
	f=" ".join(f.split())
	isPos=["Negative"]
	FullData=FullData.append(pd.DataFrame({'isPos':isPos, 'text':f}))
# Save to file
FullData.to_csv("Full_Data.csv")
######################### end of preprocess#######################################################
##################################################################################################
# Task 1: Logistic regression
# The stretagy, sklear provide the function to extract features (tokens or bigram).
# Feed a logistic regression with these features. 
# Then sklearn produce 10 fold cross validation. 
# Print out all the accuracy (10 accuracy values), calculate the mean.
##################################################################################################
# I imported from the file again
FullData=pd.read_csv("/home/admin1/Documents/Machine_learning/assignment3/Full_Data.csv")
FullData=FullData.drop(FullData.columns[[0]], axis=1)
# take the class column (Positive, Negative) as Y and the text as X. 
X=FullData['text']
Y=FullData['isPos']
######################### Sklearn ##########################################
# run unigram
Unigram_vectorizer = CountVectorizer(ngram_range=(1,1))
X_counts = Unigram_vectorizer.fit_transform(X)
LRModel = LogisticRegression()
LRModel_accuracy = cross_val_score(LRModel, X_counts, Y, cv=10, scoring='accuracy')
Task1_Unigram_Ans=np.sum(LRModel_accuracy)/10
print(Task1_Unigram_Ans)
0.8435
# run bigram
Bigram_vectorizer = CountVectorizer(ngram_range=(2,2))
X_counts = Bigram_vectorizer.fit_transform(X)
LRModel = LogisticRegression()
LRModel_accuracy = cross_val_score(LRModel, X_counts, Y, cv=10, scoring='accuracy')
Task1_Bigram_Ans=np.sum(LRModel_accuracy)/10
print(Task1_Bigram_Ans)
0.8195
############# Kares : follow tips from exercise 5 ###############################
def train_test(model, x, y, folds):
    """ This function trains and tests a Keras model with k-fold cv. 
        'folds' is the array returned by sklearn *KFold splits.
    """
    acc_sum = 0
    for trn_i, test_i in folds:
        model.fit(x[trn_i], y[trn_i], nb_epoch=1)
        _ , acc = model.evaluate(x[test_i], y[test_i])
        acc_sum += acc
    return acc_sum/len(folds)
    
Unigram_vectorizer = CountVectorizer(ngram_range=(1,1))
X_counts=Unigram_vectorizer.fit_transform(X).toarray()
X_matrix=np.asmatrix(X_counts)
y = np.zeros((len(X), 1))
for i in range(len(X)):
	if i < 1000:
		y[i]=[True] # positive
	else:
		y[i]=[False] # negative
Y_matrix=np.asmatrix(y)
single_layer = Sequential()
single_layer.add(Dense(output_dim=1, input_dim=X_matrix.shape[1], activation='sigmoid'))
single_layer.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
folds = StratifiedKFold(Y, 10)
single_layer_accuracy = train_test(single_layer, X_matrix, Y_matrix, folds)
print(single_layer_accuracy.mean())
0.6285
##########This result is suspecious, because there is a huge difference between sklearn and keras ########


