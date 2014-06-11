
from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl
import wordcloud



#from sklearn.datasets import fetch_20newsgroups
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.feature_extraction.text import HashingVectorizer
#from sklearn.feature_selection import SelectKBest, chi2
#from sklearn.linear_model import RidgeClassifier
#from sklearn.linear_model import LogisticRegression
#from sklearn.svm import LinearSVC
#from sklearn.svm import SVC
#from sklearn.linear_model import SGDClassifier
#from sklearn.linear_model import Perceptron
#from sklearn.linear_model import PassiveAggressiveClassifier
#from sklearn.naive_bayes import BernoulliNB, MultinomialNB
#from sklearn.neighbors import KNeighborsClassifier
#from sklearn.neighbors import NearestCentroid
#from sklearn.utils.extmath import density
#from sklearn import metrics
#from sklearn.cross_validation import train_test_split
#from sklearn.grid_search import GridSearchCV
#from sklearn.metrics import classification_report
#

# Sourabh : Additional imports for KDD 2014 dataset  
import string
import unicodedata
import pandas as pd
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from sklearn.datasets.base import Bunch
from sklearn.utils import check_random_state
from pprint import pprint
from sklearn.externals import joblib
import os
import sklearn
import csv
#from pytagcloud import create_tag_image, make_tags
#from pytagcloud.lang.counter import get_tag_counts

print(sklearn.__version__)


################################################################################
#  Read the dataset
################################################################################

print("Reading dataset ...")

def pklfl(x):
    dirpath = 'pkl'
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath + os.sep + os.path.relpath(__file__) + '_' + x + '.pkl' 

oper = 'calculate'
#oper = 'load'

tload = time()
if oper == 'calculate':

    labels_csv_df = pd.read_csv('../../dataset/outcomes.csv')
    labels_df = pd.DataFrame(labels_csv_df,columns=['projectid','is_exciting'])
    num_train = labels_df.shape[0]

    essays_csv_df = pd.read_csv('../../dataset/essays.csv')
    essays_df = pd.DataFrame(essays_csv_df,columns=['projectid','title'])
    num_total = essays_df.shape[0]   # ideally should be taken form projects
    num_test  = num_total - num_train

    data_df = essays_df.join(labels_df,how='outer',lsuffix='_essays_csv',rsuffix='_labels_csv')
    joblib.dump(value=labels_df,filename=pklfl('labels_df'),compress=0,cache_size=100)
    pprint(pklfl('labels_df'))
    joblib.dump(value=essays_df,filename=pklfl('essays_df'),compress=0,cache_size=100)
    pprint(pklfl('essays_df'))
    joblib.dump(value=data_df,filename=pklfl('data_df'),compress=0,cache_size=100)
    pprint(pklfl('data_df'))
else:
    pprint('Loading ...')
    pprint(pklfl('labels_df'))
    labels_df = joblib.load(filename=pklfl('labels_df'))
    pprint(pklfl('essays_df'))
    essays_df = joblib.load(filename=pklfl('essays_df'))
    pprint(pklfl('data_df'))
    data_df = joblib.load(filename=pklfl('data_df'))
    pprint('done')
    num_train = labels_df.shape[0]
    num_total = essays_df.shape[0]   # ideally should be taken form prorojects
    num_test  = num_total - num_train

tloadend = time()

print('Numbers: Train: %d, Test: %d, Total: %d' % (num_train, num_test, num_total))  
print('Time taken for load : %s : %.2f\n' % (oper,(tloadend-tload)))

#data_test_df = data_df[data_df['is_exciting'].isnull()]
#data_train_df = data_df[data_df['is_exciting'].notnull()]

#pprint(data_train_df)
#pprint(data_test_df)

# Get an instance of lemmatizer and stemmer
lmtzr = WordNetLemmatizer()
stemmer = SnowballStemmer('english')

# Allocate data structure to be fed to classifiers
txtfeatWordList = np.empty(num_total,dtype='object')
label = np.zeros(num_total,dtype='int')
train_idx = np.zeros(num_train,dtype='int')
test_idx = np.zeros(num_test,dtype='int')
train_projid = np.empty(num_train,dtype='object')
test_projid = np.empty(num_test,dtype='object')

#for title in data_df['title'].fillna(''):
train_idx_itr = 0
test_idx_itr = 0

t1 = time()
txtfeat_vec = data_df['title'].fillna('')
label_vec = data_df['is_exciting']
featid_vec = data_df['projectid_essays_csv']
t2 = time()
print(txtfeat_vec)
print()
print(label_vec)
print()
print(featid_vec)
print()
print('Time taken for column selection: %.2f seconds' % (t2-t1))


t3 = time()
if oper == 'calculate':
    for i in range(num_total):

        txtfeat = txtfeat_vec[i]
        if label_vec[i] == 't':
            lbl = 1
            train_idx[train_idx_itr] = i  # TODO: base this decision on some other criteria
            train_projid[train_idx_itr] = featid_vec[i]
            train_idx_itr = train_idx_itr + 1 
        elif label_vec[i] == 'f':
            lbl = 0
            train_idx[train_idx_itr] = i  # TODO: base this decision on some other criteria
            train_projid[train_idx_itr] = featid_vec[i]
            train_idx_itr = train_idx_itr + 1 
        else:
            lbl = 0
            test_idx[test_idx_itr] = i  # TODO: base this decision on some other criteria
            test_projid[test_idx_itr] = featid_vec[i]
            test_idx_itr = test_idx_itr + 1 
        label[i] = lbl
        
        # Create word list
        txtfeatWords = txtfeat.decode('utf-8','ignore')

        # tokenize 
        txtfeatWords2 = wordpunct_tokenize(txtfeatWords)

        # convert to lowercase
        txtfeatWords3 = [w.lower() for w in txtfeatWords2]

        # remove stop words
        txtfeatWords4 = [ word for word in txtfeatWords3 if word not in set(stopwords.words('english')) ]
        
        # lemmatize based on WordNet 
        txtfeatWords5 = [ lmtzr.lemmatize(word) for word in txtfeatWords4 ]
        
        # stem using snowball stemmer
        txtfeatWords6 = [ stemmer.stem(word) for word in txtfeatWords5 ]
        
        # remove punctuations
        txtfeatWords7 = [ word.encode('utf-8').translate(None,string.punctuation) for word in txtfeatWords6 ]
        
        # remove empty strings
        txtfeatWords8 = [ word for word in txtfeatWords7 if word <> '' ]
        
        txtfeatWordList[i] = ' '.join(txtfeatWords8)
        
        #pprint('Iteration: %d'% i)
        #pprint(txtfeatWordList[i])
   

pprint(txtfeatWordList)

text = '\n'.join([ str(txtfeatWordList[i]) for i in range(num_total) ])

#tags = make_tags(get_tag_counts( '\n'.join([ str(txtfeatWordList[i]) for i in range(num_total) ]) ))
#create_tag_image(tags, 'cloud_large.png', size=(1800, 1200), fontname='Lobster')

d = os.path.dirname(__file__)
words = wordcloud.process_text(text)
elements = wordcloud.fit_words(words)
wordcloud.draw(elements, os.path.join(d, 'lemmatized_wordle.png'))

