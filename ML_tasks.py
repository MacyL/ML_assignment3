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
# function from exercise 5
def train_test(model, x, y, folds):
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
my_single_layer = Sequential()
my_single_layer.add(Dense(output_dim=1, input_dim=X_matrix.shape[1], activation='sigmoid'))
my_single_layer.compile(loss='binary_crossentropy',optimizer='sgd', metrics=['accuracy'])
# 10 fold cross validation
folds = StratifiedKFold(Y, 10)
my_single_layer_accuracy = train_test(my_single_layer, X_matrix, Y_matrix, folds)
print(single_layer_accuracy.mean())
0.6285
# Bigram, 'memory error' during traforming to array, stop using bigram so far. 
##########This result is suspecious, because there is a huge difference between sklearn and keras ########
################################## clear all the histroy, and then move on to task 2 #####################
# Task 2 : MLP, word vector
# The strategy. For this task, use word vector to replace the token counts.
# The word vector downloaded Glove 6B word vector, I use 50 Dimension because my computer capacity.
# Because Sklearn hasn't release 0.8 version, I used Sknn for MLP. Also try Keras. 
###########################################################################################################
WordVector=open("/home/admin1/Documents/Machine_learning/assignment3/glove_6B/glove.6B.50d.txt").readlines()
#read in as list
myDictionary = dict()
for each in WordVector:
	each=each.split()
	key=each[0]
	vec=np.array([float(tmp) for tmp in each[1:]])
	myDictionary[key]=vec
	
FullData=pd.read_csv("/home/admin1/Documents/Machine_learning/assignment3/Full_Data.csv")
FullData=FullData.drop(FullData.columns[[0]], axis=1)
X=FullData['text']
Y=FullData['isPos']
# For this task, I try to tokenize by myself, get the data frame with texts as rows and tokens as columns
# NLTK regular expression tokenizer
tokenizer= RegexpTokenizer(r'\w+|\$[\d\.]+|\S+')
FullDoc=[]
for each in X:
	Unigram = tokenizer.tokenize(each)
	Unigram_counter=Counter(Unigram)
	df=[]
	# give the word vector some weights, not just simply replace it. 
	for k in Unigram_counter:
		if k in myDictionary:
			miniWordVector=myDictionary[k]*Unigram_counter[k]
			df +=[miniWordVector]
	WordVectorSum=np.mean(df,axis=0)
	FullDoc +=[WordVectorSum]

Y=pd.DataFrame(Y)
FullDocDataFrame=pd.DataFrame(FullDoc)
FullDocDataFrame=pd.concat([Y,FullDocDataFrame],axis=1)
# output to csv file so I can start from here in the future
FullDocDataFrame.to_csv("Task2_Data_Unigram.csv")
# Bigram
Full100D=[]
for each in X:
	Unigram = tokenizer.tokenize(each)
	Bigram = list(ngrams(Unigram,2))
	#The strategy is split the bigram into two keys, build their own vector sets calculate average and then concate the final vectors together. 
	Bigramkey1=[i[0] for i in Bigram]
	BigramKey1Count=Counter(Bigramkey1)
	BigramKey1minivec=[myDictionary[k]*BigramKey1Count[k] for k in BigramKey1Count if k in myDictionary]
	BigramKey1mean=np.mean(BigramKey1minivec,axis=0)	
	# key2 
	BigramKey2=[i[1] for i in Bigram]
	BigramKey2Count=Counter(BigramKey2)
	BigramKey2minivec=[myDictionary[k]*BigramKey2Count[k] for k in BigramKey2Count if k in myDictionary]
	BigramKey2mean=np.mean(BigramKey2minivec,axis=0)
	#print(myBigramKey1mean, myBigramKey2mean)
	# Concate
	BigramVec=np.concatenate((BigramKey1mean,BigramKey2mean), axis=0)
	Full100D+=[BigramVec]

Y=pd.DataFrame(Y)
Full100DDataFrame=pd.DataFrame(Full100D)
Full100DDataFrame=pd.concat([Y,Full100DDataFrame],axis=1)
Full100DDataFrame.to_csv("Task2_Data_Bigram.csv")
################################################# 
Task2=pd.read_csv("Task2_Data_Unigram.csv")
Task2=Task2.drop(Task2.columns[[0]], axis=1)
#Seperate X and Y
Y=Task2['isPos']
X=Task2[Task2.columns[1:51]]
# Transform X into matrix is very important
X_matrix = X.as_matrix()
# Set up MLP 
MLP = Classifier(
    layers=[
        Layer("Sigmoid", units=200),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=10)
    
MLP_accuracy = cross_val_score(MLP, X_matrix, Y, cv=10, scoring='accuracy')
Task2_Unigram_Ans=np.sum(MLP_accuracy)/10
print(Task2_Unigram_Ans)
0.664
# I also tried units with 50, 100, 150, 200
# The answer is 0.646, 0.6465, 0.6405
Task2=pd.read_csv("Task2_Data_Bigram.csv")
Task2=Task2.drop(Task2.columns[[0]], axis=1)
#Seperate X and Y
Y=Task2['isPos']
X=Task2[Task2.columns[1:101]]
X_matrix = X.as_matrix()
MLP = Classifier(
    layers=[
        Layer("Sigmoid", units=200),
        Layer("Softmax")],
    learning_rate=0.02,
    n_iter=10)

MLP_accuracy = cross_val_score(MLP, X_matrix, Y, cv=10, scoring='accuracy')
Task2_Bigram_Ans=np.sum(MLP_accuracy)/10
print(Task2_Bigram_Ans)
0.6725
# I also tried units with 50, 100, 150, 250
# The answer is 0.629, 0.678, 0.6375, 0.6405,
####################### clear all the history and move on to task 3 ############################
# Task 3 : Convolutional neural network
# For this task, use word vector to replace each token. 
################################################################################################
Data=pd.read_csv("/home/admin1/Documents/Machine_learning/assignment3/Full_Data.csv")
Data=Data.drop(Data.columns[[0]], axis=1)
X=Data['text']
Y=Data['isPos']



