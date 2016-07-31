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
from numpy import array,reshape
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split,KFold
from sklearn.linear_model import LogisticRegression
from sknn.mlp import Classifier, Layer
from keras.preprocessing.text import Tokenizer
from keras.layers import Embedding,Input, Dense,Dropout, Activation, Flatten,Convolution1D, MaxPooling1D
from keras.layers import Reshape, Convolution2D, MaxPooling2D
from keras.models import Sequential,Model
from keras.preprocessing import sequence
import keras.preprocessing.sequence as ks
from keras.preprocessing.sequence import pad_sequences
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
################################## end of task 1 ##########################################################
# Task 2 : MLP, word vector
# The strategy. For this task, use word vector to replace the token counts.
# The word vector downloaded Glove 6B word vector, I use 50 Dimension because my computer capacity.
# Because Sklearn hasn't release 0.8 version, I used Sknn for MLP. I also tried Keras, but it couldn't work for me. 
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
############################# end of task 2 ###################################################
# Task 3 : Convolutional neural network
# For this task, use word vector to replace each token. 
################################################################################################
Data=pd.read_csv("/home/admin1/Documents/Machine_learning/assignment3/Full_Data.csv")
Data=Data.drop(Data.columns[[0]], axis=1)
X=Data['text']
Y=Data['isPos']

# Clean the data, remove 's, 'nt, 've
# Code reference : https://github.com/yoonkim/CNN_sentence
def clean_string(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Every dataset is lower cased except for TREC
    """     
    string = re.sub(r"\'s", " \'s", string) 
    string = re.sub(r"\'ve", " \'ve", string) 
    string = re.sub(r"n\'t", " n\'t", string) 
    string = re.sub(r"\'re", " \'re", string) 
    string = re.sub(r"\'d", " \'d", string) 
    string = re.sub(r"\'ll", " \'ll", string) 
    string = re.sub(r",", " , ", string) 
    string = re.sub(r"!", " ! ", string) 
    string = re.sub(r"\(", " \( ", string) 
    string = re.sub(r"\)", " \) ", string) 
    string = re.sub(r"\?", " \? ", string)  
    return string.strip() 

myTexts=[]
for each in X:
	myEach=clean_string(each)
	myTexts +=[myEach]
	
# tokenize texts into tokens 
tokenizer = Tokenizer(nb_words=800)
tokenizer.fit_on_texts(myTexts)
sequences = tokenizer.texts_to_sequences(myTexts)
word_index = tokenizer.word_index
# trim the length of each sequence to the same length, I set 300.
data = pad_sequences(sequences, maxlen=300)
y = np.zeros((len(myTexts), 1))

for i in range(len(myTexts)):
	if i < 1000:
		y[i]=[True] # positive
	else:
		y[i]=[False] # negative

embedding_matrix = np.zeros((len(word_index) + 1, 50))
for word, i in word_index.items():
	embedding_vector = myDictionary.get(key)
	if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        	embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            50,
                            weights=[embedding_matrix],
                            input_length=300,
                            trainable=False)


# first trail, purely just word sequence.
scores_conv = []
kf_total = KFold(len(data), n_folds=10, shuffle=True, random_state=3)
for train_index, test_index in kf_total:
	myTrain=data[train_index]
	myTrainResponse=y[train_index]
	myTest=data[test_index]
	expected=y[test_index]
	model = Sequential()
	model.add(Embedding(len(word_index) + 1,50,input_length=300,dropout=0.2))
	model.add(Convolution1D(nb_filter=200,filter_length=5,border_mode='valid',activation='relu',subsample_length=1))
	model.add(MaxPooling1D(pool_length=model.output_shape[1]))
	model.add(Flatten())
	model.add(Dense(200))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(myTrain, myTrainResponse,batch_size=200, nb_epoch=20)
	score = model.evaluate(myTest, expected)
	scores_conv.append(score[1])
	
with open('task3_output.txt', 'wb') as f:
    pickle.dump(scores_conv, f)
# If want to load the data. 
with open('task3_output.txt', 'rb') as f:
    a = pickle.load(f)
np.mean(a)
0.768


# second trail. 
scores_conv2 = []
kf_total = KFold(len(data), n_folds=10, shuffle=True, random_state=3)
for train_index, test_index in kf_total:
	myTrain=data[train_index]
	myTrainResponse=y[train_index]
	myTest=data[test_index]
	expected=y[test_index]
	model = Sequential()
	model.add(Embedding(len(word_index) + 1,50,input_length=300,dropout=0.2))
	model.add(Convolution1D(nb_filter=200,filter_length=5,border_mode='valid',activation='relu',subsample_length=2))
	model.add(MaxPooling1D(pool_length=model.output_shape[1]))
	model.add(Flatten())
	model.add(Dense(250))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(150))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(myTrain, myTrainResponse,batch_size=200, nb_epoch=20)
	score = model.evaluate(myTest, expected)
	scores_conv2.append(score[1])

np.mean(scores_conv)
1
# I didn't expect this, it might due to overfitting. Also, I wordered what happened to my code. 

# Third trail 
WordVector=open("/home/admin1/Documents/Machine_learning/assignment3/glove_6B/glove.6B.50d.txt").readlines()
#read in as list
myDictionary = dict()
for each in WordVector:
	each=each.split()
	key=each[0]
	vec=np.array([float(tmp) for tmp in each[1:]])
	myDictionary[key]=vec

# create an numpy ndarray
X_CNN = np.zeros((len(data),300,embedding_dim))

scores_conv5 = []
kf_total = KFold(len(data), n_folds=10, shuffle=True, random_state=3)
for train_index, test_index in kf_total:
	myTrain=data[train_index]
	myTrainResponse=y[train_index]
	myTest=data[test_index]
	expected=y[test_index]
	model = Sequential()
	model.add(Embedding(len(word_index) + 1,50,input_length=300, weights=[embedding_matrix],dropout=0.2))
	model.add(Convolution1D(nb_filter=200,filter_length=5,border_mode='valid',activation='relu',subsample_length=2))
	model.add(MaxPooling1D(pool_length=model.output_shape[1]))
	model.add(Flatten())
	model.add(Dense(200))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(myTrain, myTrainResponse,batch_size=200, nb_epoch=20)
	score = model.evaluate(myTest, expected)
	scores_conv5.append(score[1])

np.mean(scores_conv5)
0.50
# This model does not perform well, but I have my word vector input as weighting matrix, I supposed this should work perfectly. 
# Fourth trail 
# Input one more layer 

scores_conv6 = []
kf_total = KFold(len(data), n_folds=10, shuffle=True, random_state=3)
for train_index, test_index in kf_total:
	myTrain=data[train_index]
	myTrainResponse=y[train_index]
	myTest=data[test_index]
	expected=y[test_index]
	model = Sequential()
	model.add(Embedding(len(word_index) + 1,50,input_length=300, weights=[embedding_matrix],dropout=0.2))
	model.add(Convolution1D(nb_filter=200,filter_length=10,border_mode='valid',activation='relu',subsample_length=1))
	model.add(Convolution1D(nb_filter=100,filter_length=5,border_mode='valid',activation='relu',subsample_length=2))
	model.add(MaxPooling1D(pool_length=model.output_shape[1]))	
	model.add(Flatten())
	model.add(Dense(200))
	model.add(Dropout(0.2))
	model.add(Activation('relu'))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(myTrain, myTrainResponse,batch_size=200, nb_epoch=5)
	score = model.evaluate(myTest, expected)
	scores_conv6.append(score[1])
# For this model, the nb_epoch ran to the second time and then the accuracy reached 1. So I reduced the nb_epoch number to 5. 
np.mean(scores_conv6)
1


# Fifth time trail.  This model require the input as a four dimensions ndarray
# But sadly, my computer crashes when everytime I try to run this model 
data = pad_sequences(sequences, maxlen=150)

X_CNN = np.zeros((len(data), 150,50))
# This code idea came from Marlon Betz
# Here, I have my word vector dictionary created and my data already tranformed to numric sequence. 
# Hence, this part is not exactly the same like his code.   
for i in range(len(data)):
	oneData = data[i]
	token_index = 0
	for t in oneData:
		X_CNN[i,token_index] = embedding_matrix[t]
	token_index +=1

# Embedding
max_features = 40023
maxlen = 150
embedding_size = 50
nb_feature_maps=25

# Convolution
filter_length = 5
nb_filter = 200
pool_length = 4
bi_gram=2
tri_gram=3
batch_size=200
nb_epoch=2
nb_pool=2

scores_conv7 = []
kf_total = KFold(len(X_CNN), n_folds=10, shuffle=True, random_state=3)
for train_index, test_index in kf_total:
	myTrain=X_CNN[train_index]
	myTrainResponse=y[train_index]
	myTest=X_CNN[test_index]
	expected=y[test_index]
	model = Sequential()
	model.add(Convolution2D(nb_filter,tri_gram, tri_gram, border_mode='valid', input_shape= (1, maxlen, embedding_size)))
	model.add(Activation('relu'))
	model.add(Convolution2D(nb_filter,bi_gram,bi_gram))
	model.add(Activation('relu'))
	model.add(MaxPooling2D(pool_size=(2,2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(100))
	model.add(Activation('relu'))
	model.add(Dropout(0.2))
	model.add(Dense(1))
	model.add(Activation('sigmoid'))
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
	model.fit(myTrain, myTrainResponse.reshape(-1,1,maxlen,embedding_size),batch_size=200, nb_epoch=5)
	score = model.evaluate(myTest.reshape(-1,1,maxlen,embedding_size), expected)
	scores_conv7.append(score[1])
	
# I have tried hard to make it work, but my computer kept crashing. so I have no result from this CNN model. 
# I wonder it is due to my computer capacity or the code has serious problem in it. 
