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
MyBigFile=pd.DataFrame()
for name in filesPos:
	f=open(name,'r').read()
	f=re.sub(r'\d+','',f)
	# Remove duplicated white space.
	f=" ".join(f.split())
	isPos=["Positive"]
	MyBigFile=MyBigFile.append(pd.DataFrame({'isPos':isPos, 'text':f}))

for names in filesNeg:
	f=open(names,'r').read()
	f=re.sub(r'\d+','',f)
	# Remove duplicated white space.
	f=" ".join(f.split())
	isPos=["Negative"]
	MyBigFile=MyBigFile.append(pd.DataFrame({'isPos':isPos, 'text':f}))
# Save to file
MyBigFile.to_csv("My_Big_File.csv")

