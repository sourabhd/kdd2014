from __future__ import print_function
import pandas as pd
import numpy as np
import pdb
import sys
import re
import string
from time import time
import scipy.io as sio
from sklearn.preprocessing import LabelEncoder

def tolower(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower().decode('utf-8')
        except:
            return " ".join(re.findall(r'\w+', "",flags = re.UNICODE | re.LOCALE)).lower().decode('utf-8')

def removepunct(s):
    try:
        return ''.join([ word.encode('utf-8').translate(None,string.punctuation) for word in s ])
    except:
        return ''

def main():
#   Read all data
    essays = pd.read_csv(open('../../dataset/essays.csv','r'))
    projects = pd.read_csv(open('../../dataset/projects.csv','r'))
    outcomes = pd.read_csv(open('../../dataset/outcomes.csv','r'))

    essays_c = essays[['projectid','title','essay']].sort(column='projectid')
    projects_c = projects[['projectid', 'date_posted']].sort(column='projectid')
    outcomes_c = outcomes[['projectid','is_exciting']].sort(column='projectid')

    ep = pd.merge(projects_c, essays_c, on='projectid')
    epo = pd.merge(ep, outcomes_c, on='projectid', how='outer')

#    Cleanup essays text data 
    epo['essay_cl'] = epo['essay'].apply(tolower).apply(removepunct)
    epo_arr = np.array(epo['essay_cl'],dtype='string')
    le = LabelEncoder()
    le.fit(['t','f', ''])
    labels = le.transform(epo['is_exciting'].fillna(''))
    train_idx = np.where(epo['date_posted'] < '2014-01-01')[0]
    test_idx = np.where(epo['date_posted'] >= '2014-01-01')[0]

    spl = re.compile('\s+')
    WV = {}
    f = open('data/embeddings/word2vec_defaultsettings_vectors.txt','r')
    L = f.readline().strip().split()
    num_words = int(L[0])
    wv_dim = int(L[1])
    print("Num Words: %d, Word dimension: %d" % (num_words,wv_dim))
    words = np.empty(shape=(num_words,1),dtype = 'object')
    k = 0
    for line in f:
        #print("Line: " + line)
        L = re.split(spl, line.strip()) 
        WV[str(L[0])] = np.array([float(x) for x in L[1:len(L)]], dtype = float)
        words[k,:] = str(L[0])
        k = k + 1
    f.close()
    #print(words)

    FV = np.zeros((epo_arr.shape[0],len(WV.keys())), dtype = float)
    i = 0
    for ess in epo_arr:  # i loop 
        #print(ess)
        j = 0
        for w in WV.keys():   # j loop
            FV[i,j] = np.inf    
            ess_s = set(re.split(spl,ess))
            for wd in ess_s: 
                try:
                    #print("wd: %s\tw: %s" % (wd,w))
                    d = np.linalg.norm(WV[wd] - WV[w])
                    if d < FV[i,j]:
                        FV[i,j] = d   
                except:
                    pass
            j = j + 1
        i = i + 1
        if i % 10000 == 0:
            print("Iteration : %d / %d" % (i,epo_arr.shape[0]))
        #if i > 50:
        #    break
            
    FV = np.exp(-1.0 * FV)

    #np.save('FV', FV)
    sio.savemat('WV', {'WV_FV':FV, 'WV_Labels':labels, 'WV_tr_idx':train_idx, 'WV_te_idx':test_idx, 'WV_words':words})

if __name__ == '__main__':
    start_time = time()
    try:
        main()
        end_time = time()
        print("Successful termination\n")
        print("Time taken: %.2f\n" % (end_time-start_time))
    except:
        end_time = time()
        print("Unsuccessful termination\n")
        print("Time taken: %.2f\n" % (end_time-start_time))
        pdb.post_mortem(sys.exc_info()[2])

