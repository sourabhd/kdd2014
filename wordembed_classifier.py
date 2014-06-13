from __future__ import print_function 
import platform
import sklearn
import pandas as pd
from time import time
#from pprint import pprint
import sys
#import os
import pdb
import numpy as np
import string
from nltk import wordpunct_tokenize
from nltk.corpus import stopwords
#from nltk.stem.wordnet import WordNetLemmatizer
#from nltk.stem.snowball import SnowballStemmer
from datetime import datetime
from sklearn.utils import check_random_state
from sklearn.datasets.base import Bunch

oper = 'calculate'

def print_versions():
    # print version info
    print("Python: " + platform.python_version())
    print("Sklearn: " + sklearn.__version__)
    print("Pandas: " + pd.__version__)


def extract_words():

    t0 = time()
    # Read projects 
    projects_csv_df = pd.read_csv('../../dataset/projects.csv')
    projects_df = pd.DataFrame(projects_csv_df,columns=['projectid','date_posted'])
    t1 = time()
    print("projects.csv loaded in %.2f sec" % (t1-t0))

    # Read outcomes
    t2 = time()
    labels_csv_df = pd.read_csv('../../dataset/outcomes.csv')
    labels_df = pd.DataFrame(labels_csv_df,columns=['projectid','is_exciting'])
    t3 = time()
    print("outcomes.csv loaded in %.2f sec" % (t3-t2))

    # Read essays
    t4 = time()
    essays_csv_df = pd.read_csv('../../dataset/essays.csv')
    essays_df = pd.DataFrame(essays_csv_df,columns=['projectid','title'])
    essay_proj_df = pd.merge(projects_df, essays_df, on='projectid')
    t5 = time()
    print("essays.csv loaded in %.2f sec" % (t5-t4))

    # Calculate numbers
    num_train = labels_df.shape[0]
    num_total = projects_df.shape[0]   
    num_test  = num_total - num_train

    data_df = essay_proj_df.join(labels_df,how='outer',lsuffix='_essays_csv',rsuffix='_labels_csv')

    # Get an instance of lemmatizer and stemmer
    #lmtzr = WordNetLemmatizer()
    #stemmer = SnowballStemmer('english')

    t6 = time()
    # Allocate data structure to be fed to classifiers
    txtfeatWordList = np.empty(num_total,dtype='object')
    label = np.zeros(num_total,dtype='int')
    train_idx = np.zeros(num_train,dtype='int')
    test_idx = np.zeros(num_test,dtype='int')
    train_projid = np.empty(num_train,dtype='object')
    test_projid = np.empty(num_test,dtype='object')

    train_idx_itr = 0
    test_idx_itr = 0

    txtfeat_vec = data_df['title'].fillna('')
    label_vec = data_df['is_exciting']
    featid_vec = data_df['projectid_essays_csv']
    projects_date_vec = data_df['date_posted'].apply(pd.to_datetime).apply(datetime.date)

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
            #txtfeatWords5 = [ lmtzr.lemmatize(word) for word in txtfeatWords4 ]

            # stem using snowball stemmer
            #txtfeatWords6 = [ stemmer.stem(word) for word in txtfeatWords5 ]

            # switched off stemming and lemmatization
            txtfeatWords6 = txtfeatWords4

            # remove punctuations
            txtfeatWords7 = [ word.encode('utf-8').translate(None,string.punctuation) for word in txtfeatWords6 ]

            # remove empty strings
            txtfeatWords8 = [ word for word in txtfeatWords7 if word <> '' ]

            txtfeatWordList[i] = ' '.join(txtfeatWords8)

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

    t7 =time()
    print('Time taken for converting to sklearn: %.2f seconds' % (t7-t6))


    print(data_train)
    print(data_test)
    print('data loaded')


def main():
    print_versions()
    extract_words()

if __name__ == '__main__':
    try:
        main()
    except:
        pdb.post_mortem(sys.exc_info()[2])

