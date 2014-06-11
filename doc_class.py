"""
======================================================
Classification of text documents using sparse features
======================================================

This is an example showing how scikit-learn can be used to classify documents
by topics using a bag-of-words approach. This example uses a scipy.sparse
matrix to store the features and demonstrates various classifiers that can
efficiently handle sparse matrices.

The dataset used in this example is the 20 newsgroups dataset. It will be
automatically downloaded, then cached.

The bar plot indicates the accuracy, training time (normalized) and test time
(normalized) of each classifier.

"""

# Author: Peter Prettenhofer <peter.prettenhofer@gmail.com>
#         Olivier Grisel <olivier.grisel@ensta.org>
#         Mathieu Blondel <mathieu@mblondel.org>
#         Lars Buitinck <L.J.Buitinck@uva.nl>
# License: BSD 3 clause

from __future__ import print_function

import logging
import numpy as np
from optparse import OptionParser
import sys
from time import time
import pylab as pl

#from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.utils.extmath import density
from sklearn import metrics
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from datetime import datetime
from datetime import date

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
import platform

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')


# parse commandline arguments
op = OptionParser()
op.add_option("--report",
              action="store_true", dest="print_report",
              help="Print a detailed classification report.")
op.add_option("--chi2_select",
              action="store", type="int", dest="select_chi2",
              help="Select some number of features using a chi-squared test")
op.add_option("--confusion_matrix",
              action="store_true", dest="print_cm",
              help="Print the confusion matrix.")
op.add_option("--top10",
              action="store_true", dest="print_top10",
              help="Print ten most discriminative terms per class"
                   " for every classifier.")
op.add_option("--all_categories",
              action="store_true", dest="all_categories",
              help="Whether to use all categories or not.")
op.add_option("--use_hashing",
              action="store_true",
              help="Use a hashing vectorizer.")
op.add_option("--n_features",
              action="store", type=int, default=2 ** 16,
              help="n_features when using the hashing vectorizer.")
op.add_option("--filtered",
              action="store_true",
              help="Remove newsgroup information that is easily overfit: "
                   "headers, signatures, and quoting.")

(opts, args) = op.parse_args()
if len(args) > 0:
    op.error("this script takes no arguments.")
    sys.exit(1)

print(__doc__)
op.print_help()
print()


# print version info
print("Python: " + platform.python_version())
print("Sklearn: " + sklearn.__version__)
print("Pandas: " + pd.__version__)

###############################################################################
# Load some categories from the training set
#if opts.all_categories:
#    categories = None
#else:
#    categories = [
#        'alt.atheism',
#        'talk.religion.misc',
#        'comp.graphics',
#        'sci.space',
#    ]
#
#if opts.filtered:
#    remove = ('headers', 'footers', 'quotes')
#else:
#    remove = ()
#
#print("Loading 20 newsgroups dataset for categories:")
#print(categories if categories else "all")

#data_train = fetch_20newsgroups(subset='train', categories=categories,
#                                shuffle=True, random_state=42,
#                                remove=remove)

#data_test = fetch_20newsgroups(subset='test', categories=categories,
#                               shuffle=True, random_state=42,
#                               remove=remove)

# Incompatibilities in version
#data_train = fetch_20newsgroups(subset='train', categories=categories,
#                                shuffle=True, random_state=42
#                                )
#
#data_test = fetch_20newsgroups(subset='test', categories=categories,
#                               shuffle=True, random_state=42
#                               )

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

    projects_csv_df = pd.read_csv('../../dataset/projects.csv')
    projects_df = pd.DataFrame(projects_csv_df,columns=['projectid','date_posted'])
    
    labels_csv_df = pd.read_csv('../../dataset/outcomes.csv')
    labels_df = pd.DataFrame(labels_csv_df,columns=['projectid','is_exciting'])

    essays_csv_df = pd.read_csv('../../dataset/essays.csv')
    essays_df = pd.DataFrame(essays_csv_df,columns=['projectid','title'])
    essay_proj_df = pd.merge(projects_df, essays_df, on='projectid')
    num_train = labels_df.shape[0]
    num_total = projects_df.shape[0]   
    num_test  = num_total - num_train

    data_df = essay_proj_df.join(labels_df,how='outer',lsuffix='_essays_csv',rsuffix='_labels_csv')

    joblib.dump(value=projects_csv_df,filename=pklfl('projects_csv_df'),compress=0,cache_size=100)
    pprint(pklfl('projects_csv_df'))
    joblib.dump(value=labels_df,filename=pklfl('essay_proj_df'),compress=0,cache_size=100)
    pprint(pklfl('essay_proj_df'))
    joblib.dump(value=essays_df,filename=pklfl('essays_df'),compress=0,cache_size=100)
    pprint(pklfl('essays_df'))
    joblib.dump(value=data_df,filename=pklfl('data_df'),compress=0,cache_size=100)
    pprint(pklfl('data_df'))
else:
    pprint('Loading ...')
    pprint(pklfl('projects_csv_df'))
    projects_csv_df = joblib.load(filename=pklfl('projects_csv_df'))
    pprint(pklfl('essay_proj_df'))
    essay_proj_df = joblib.load(filename=pklfl('essay_proj_df'))
    pprint(pklfl('essays_df'))
    essays_df = joblib.load(filename=pklfl('essays_df'))
    pprint(pklfl('data_df'))
    data_df = joblib.load(filename=pklfl('data_df'))
    pprint('done')
    num_train = labels_df.shape[0]
    num_total = projects_df.shape[0]   
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
projects_date_vec = data_df['date_posted'].apply(pd.to_datetime).apply(datetime.date)
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
        if projects_date_vec[i] < datetime.strptime('2014-01-01', '%Y-%m-%d').date():
            if label_vec[i] == 't':
                lbl = 1
                train_idx[train_idx_itr] = i  
                train_projid[train_idx_itr] = featid_vec[i]
                train_idx_itr = train_idx_itr + 1
            #elif label_vec[i] == 'f':
            else:
                lbl = 0
                train_idx[train_idx_itr] = i  
                train_projid[train_idx_itr] = featid_vec[i]
                train_idx_itr = train_idx_itr + 1
        else:
            lbl = 0
            test_idx[test_idx_itr] = i  
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


    joblib.dump(txtfeatWordList,pklfl('txtfeatWordList') )
    joblib.dump(label,pklfl('label'))
    joblib.dump(train_idx,pklfl('train_idx'))
    joblib.dump(test_idx,pklfl('test_idx'))
    joblib.dump(train_projid,pklfl('train_projid'))
    joblib.dump(test_projid,pklfl('test_projid'))
else:
    txtfeatWordList = joblib.load(pklfl('txtfeatWordList'))
    label = joblib.load(pklfl('label'))
    train_idx = joblib.load(pklfl('train_idx'))
    test_idx = joblib.load(pklfl('test_idx'))
    train_projid = joblib.load(pklfl('train_projid'))
    test_projid = joblib.load(pklfl('test_projid'))

t4 = time()
print('Time taken for %s (for loop): %.2f seconds' % (oper,(t4-t3)))

t5 = time()
train_idx_sfl = train_idx.copy()
random_state = 42
shuffle = False
if shuffle:
         print('Shuffling training data')
         random_state = check_random_state(random_state)
         random_state.shuffle(train_idx_sfl)

descr = '2014 KDD Cup dataset'
data_train =  Bunch(data=txtfeatWordList[train_idx_sfl], target=label[train_idx_sfl].astype(np.int), target_names=np.arange(2), images=None, DESCR=descr)
data_test =  Bunch(data=txtfeatWordList[test_idx], target=label[test_idx].astype(np.int), target_names=np.arange(2), images=None, DESCR=descr)

t6 =time()
print('Time taken for converting to sklearn: %.2f seconds' % (t6-t5))


print(data_train)
print(data_test)
print('data loaded')

print('Train Shape:: %d %d :: %d' % (data_train.data.shape[0], data_train.data.shape[0], data_train.target.shape[0]))
print('Test Shape:: %d %d :: %d' % (data_test.data.shape[0], data_test.data.shape[0], data_test.target.shape[0]))

#categories = data_train.target_names    # for case categories == None
categories = [0 , 1]

def size_mb(docs):
    return sum(len(s.decode('utf-8')) for s in docs) / 1e6

data_train_size_mb = size_mb(data_train.data)
data_test_size_mb = size_mb(data_test.data)
#
print("%d documents - %0.3fMB (training set)" % (
    len(data_train.data), data_train_size_mb))
print("%d documents - %0.3fMB (test set)" % (
    len(data_test.data), data_test_size_mb))
#print("%d categories" % len(categories))
print()

# split a training set and a test set
y_train, y_test = data_train.target, data_test.target

print("Extracting features from the training dataset using a sparse vectorizer")
t0 = time()
if opts.use_hashing:
    vectorizer = HashingVectorizer(stop_words='english', non_negative=True,
                                   n_features=opts.n_features)
    X_train = vectorizer.transform(data_train.data)
else:
    vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                                 stop_words='english')
    X_train = vectorizer.fit_transform(data_train.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_train_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_train.shape)
print()

print("Extracting features from the test dataset using the same vectorizer")
t0 = time()
X_test = vectorizer.transform(data_test.data)
duration = time() - t0
print("done in %fs at %0.3fMB/s" % (duration, data_test_size_mb / duration))
print("n_samples: %d, n_features: %d" % X_test.shape)
print()

if opts.select_chi2:
    print("Extracting %d best features by a chi-squared test" %
          opts.select_chi2)
    t0 = time()
    ch2 = SelectKBest(chi2, k=opts.select_chi2)
    X_train = ch2.fit_transform(X_train, y_train)
    X_test = ch2.transform(X_test)
    print("done in %fs" % (time() - t0))
    print()


def trim(s):
    """Trim string to fit on terminal (assuming 80-column display)"""
    return s if len(s) <= 80 else s[:77] + "..."


# mapping from integer feature name to original token string
if opts.use_hashing:
    feature_names = None
else:
    feature_names = np.asarray(vectorizer.get_feature_names())


###############################################################################
# Benchmark classifiers
def benchmark(clf):

    print('_' * 80)
    print("Training: ")
    print(clf)
    t0 = time()
    clf.fit(X_train, y_train)
    train_time = time() - t0
    print("train time: %0.3fs" % train_time)

    t0 = time()
    pred = clf.predict(X_test)
    test_time = time() - t0
    print("test time:  %0.3fs" % test_time)

    print(pred)
    print()
    clf_descr = str(clf).split('(')[0]
    #return clf_descr, score, train_time, test_time
    return clf_descr, pred, train_time, test_time


results = []
#for clf, name in (
#        (RidgeClassifier(tol=1e-2, solver="lsqr"), "Ridge Classifier"),
#        (Perceptron(n_iter=50), "Perceptron"),
#        (PassiveAggressiveClassifier(n_iter=50), "Passive-Aggressive"),
#        (KNeighborsClassifier(n_neighbors=10), "kNN")):
#    print('=' * 80)
#    print(name)
#    results.append(benchmark(clf))
#
#for penalty in ["l2", "l1"]:
#    print('=' * 80)
#    print("%s penalty" % penalty.upper())
#    # Train Liblinear model
#    results.append(benchmark(LinearSVC(loss='l2', penalty=penalty,
#                                            dual=False, tol=1e-3)))
#
#    # Train SGD model
#    results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                           penalty=penalty)))
#
## Train SGD with Elastic Net penalty
#print('=' * 80)
#print("Elastic-Net penalty")
#results.append(benchmark(SGDClassifier(alpha=.0001, n_iter=50,
#                                       penalty="elasticnet")))
#
## Train NearestCentroid without threshold
#print('=' * 80)
#print("NearestCentroid (aka Rocchio classifier)")
#results.append(benchmark(NearestCentroid()))
#
# Train sparse Naive Bayes classifiers
#print('=' * 80)
#print("Naive Bayes")
#results.append(benchmark(MultinomialNB(alpha=.01)))
#results.append(benchmark(BernoulliNB(alpha=.01)))


#class L1LinearSVC(LinearSVC):
#class L1LinearSVC(LinearSVC):
class MyClassifier(LogisticRegression):

    def fit(self, X, y):
        # The smaller C, the stronger the regularization.
        # The more regularization, the more sparsity.

        print('_' * 80)
        print("Cross validation: ")

#        param_grid = [
#        {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000], 'kernel': ['linear']},
#        {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000], 'gamma': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000], 'kernel': ['rbf']},
#        ]
#        param_grid = [
#        {'C': [0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10, 100, 1000, 100000]},
#        ]

        param_grid = [
                {'C': np.logspace(-3,3,7), 'penalty':['l1','l2']},
        ]
        print(param_grid)
        scoring = 'roc_auc'
        num_folds = 5

        #X_train_val, X_test_val, y_train_val, y_test_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)

        #svc = LinearSVC(penalty="l1", dual=False, tol=1e-3)
        svc = LogisticRegression(dual=False, tol=1e-3, class_weight='auto')
        start = time()
        clf = GridSearchCV(svc, param_grid=param_grid, cv=num_folds, scoring=scoring,verbose=2,n_jobs=joblib.cpu_count())
        clf.fit(X,y)
        print(clf)
        print("GridSearchCV took %.2f seconds for %d candidate parameter settings." % (time() - start, len(clf.grid_scores_)))

        print("Grid Scores:")
        print()
        print(clf.grid_scores_)
        print()
        print("Best estimator :")
        print()
        print(clf.best_estimator_)
        print()
        print("Best score :")
        print()
        print(clf.best_score_)
        print()
        print("Best Parameters :")
        print()
        print(clf.best_params_)
        print()

#        self.transformer_ = LinearSVC(C=clf.best_estimator_.C, penalty="l1",
#                                      dual=False, tol=1e-3, verbose=2)
#        X = self.transformer_.fit_transform(X, y)
#        return LinearSVC.fit(self, X, y)

        self.transformer_ = LogisticRegression(C=clf.best_estimator_.C, penalty=clf.best_estimator_.penalty,
                                      dual=False, tol=1e-3, class_weight='auto')
        X = self.transformer_.fit_transform(X, y)
        return LogisticRegression.fit(self, X, y)


    def predict(self, X):
        X = self.transformer_.transform(X)
        #return LinearSVC.predict(self, X)
        return LogisticRegression.predict_proba(self, X)

print('=' * 80)
#print("LinearSVC with L1-based feature selection")
#results.append(benchmark(L1LinearSVC()))
print("Logistic Regression")
results.append(benchmark(MyClassifier()))


# make some plots

indices = np.arange(len(results))

results = [[x[i] for x in results] for i in range(4)]

#clf_names, score, training_time, test_time = results
clf_names, pred, training_time, test_time = results
training_time = np.array(training_time) / np.max(training_time)
test_time = np.array(test_time) / np.max(test_time)
print("Training Time : %.2f\n" % training_time)
print("Test Time : %.2f\n" % test_time)
ofilename = 'logit.csv'
print("Writing to output file %s ... " % ofilename);

try:
    ofile = open(ofilename,'w')
    ofile.write("%s,%s%s" % ('projectid','is_exciting',os.linesep))
    for i in range(num_test):
        ofile.write("%s,%.6f%s" % (test_projid[i],pred[0][i,1],os.linesep))
    ofile.close()
    print(" done\n")
    print()
except IOError:
    print(" failed\n")
    print()

#
##pl.figure(figsize=(12,8))
#pl.title("Score")
#pl.barh(indices, score, .2, label="score", color='r')
#pl.barh(indices + .3, training_time, .2, label="training time", color='g')
#pl.barh(indices + .6, test_time, .2, label="test time", color='b')
#pl.yticks(())
#pl.legend(loc='best')
#pl.subplots_adjust(left=.25)
#pl.subplots_adjust(top=.95)
#pl.subplots_adjust(bottom=.05)
#
#for i, c in zip(indices, clf_names):
#    pl.text(-.3, i, c)
#
#pl.show()
