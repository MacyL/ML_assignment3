#!/usr/bin/env python3
######################## All The library I Need ################################
import pandas as pd
import scipy as sp
import numpy as np
import theano
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
# take the class column (Positive, Negative) as Y and the text as X. 
X=FullData['text']
Y=FullData['isPos']
# extract features from X. 



