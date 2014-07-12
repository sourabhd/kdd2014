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
from sklearn.mixture import GMM
# from sklearn.preprocessing import Imputer
# from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from copy import deepcopy


def tolower(s):
        try:
            return " ".join(re.findall(r'\w+', s, flags=re.UNICODE | re.LOCALE)).lower().decode('utf-8')
        except:
            return " ".join(re.findall(r'\w+', "", flags=re.UNICODE | re.LOCALE)).lower().decode('utf-8')


def removepunct(s):
    try:
        return ''.join([word.encode('utf-8').translate(None, string.punctuation) for word in s])
    except:
        return ''


class Learner:
        def __init__(self, alldata, labels=None):
                # data = deepcopy(alldata)
                # print("00: (%d,%d)" % (data.shape[0], data.shape[1]))
                # imp = Imputer(missing_values='NaN', strategy='median', axis=0)
                # imp.fit(data)
                # self.data = deepcopy(imp.transform(data))
                self.data = deepcopy(alldata)
                self.labels = deepcopy(labels)
#                if labels is None:
#                    self.labels = None
#                else:
#                    print("0: (%d,%d)" % (self.data.shape[0], self.data.shape[1]))
#                    le = LabelEncoder()
#                    le.fit(['f', 't'])
#                    self.labels = le.transform(labels)

        def fit_and_predict(self, train_idx, test_idx, sklearn_model, params):
                sklearn_model.params = params
                print("1: (%d,%d)" % (self.data.shape[0], self.data.shape[1]))
                # print(self.data)
                sklearn_model.fit(self.data[train_idx, :], self.labels[train_idx])
                print("2: (%d,%d)" % (self.data.shape[0], self.data.shape[1]))
                print(self.data.shape)
                print(type(self.data[1:2, :]))
                print(self.data[1:2, :].shape)
                print(self.data[train_idx, :])
                print("Heree .................................................")
                print(test_idx)
                print(self.data[test_idx, :])
                print("Heree .................................................")
                try:
                        prob = sklearn_model.predict_proba(self.data[test_idx, :])
                except:
                        scores = sklearn_model.decision_function(self.data[test_idx, :])
                        prob = 1. / (1. + np.exp(-scores))
                return prob

        def fit_and_model(self, train_idx, test_idx, sklearn_model, params):
                sklearn_model.params = deepcopy(params)
                print("1: (%d,%d)" % (self.data.shape[0], self.data.shape[1]))
                print(sklearn_model.params)
                # print(self.data)
                return sklearn_model.fit(self.data[train_idx, :])


def main():
    # Read all data
    essays = pd.read_csv(open('../essays.csv', 'r'))
    projects = pd.read_csv(open('../projects.csv', 'r'))
    outcomes = pd.read_csv(open('../outcomes.csv', 'r'))

    essays_c = essays[['projectid', 'title', 'essay']].sort(column='projectid')
    projects_c = projects[['projectid', 'date_posted']].sort(column='projectid')
    outcomes_c = outcomes[['projectid', 'is_exciting']].sort(column='projectid')

    ep = pd.merge(projects_c, essays_c, on='projectid')
    epo = pd.merge(ep, outcomes_c, on='projectid', how='outer')

    # Cleanup essays text data
    epo['essay_cl'] = epo['essay'].apply(tolower).apply(removepunct)
    epo_arr = np.array(epo['essay_cl'], dtype='string')
    le = LabelEncoder()
    le.fit(['t', 'f', ''])
    labels = le.transform(epo['is_exciting'].fillna(''))
    train_idx = np.where(epo['date_posted'] < '2014-01-01')[0]
    test_idx = np.where(epo['date_posted'] >= '2014-01-01')[0]

    spl = re.compile('\s+')
    WV = {}
    f = open('data/embeddings/word2vec_defaultsettings_vectors.txt', 'r')
    L = f.readline().strip().split()
    num_words = int(L[0])
    wv_dim = int(L[1])
    print("Num Words: %d, Word dimension: %d" % (num_words, wv_dim))
    words = np.empty(shape=(num_words, 1), dtype = 'object')
    k = 0
    for line in f:
        # print("Line: " + line)
        L = re.split(spl, line.strip())
        WV[str(L[0])] = np.array([float(x) for x in L[1:]], dtype=float)
        words[k, :] = str(L[0])
        k = k + 1
    f.close()
    # print(words)

    # Learn a mixture model on words
    WV_m = np.vstack(WV.values())
    n_components = 50
    cv_type = 'full'
    n_init = 50
    gmm_params = {'n_components': n_components, 'covariance_type': cv_type, 'n_init': n_init}  
    L = Learner(WV_m, None)
    G = L.fit_and_model(range(len(WV.keys())), range(0), GMM(n_components=n_components, covariance_type=cv_type, n_init=n_init), gmm_params)

    print("GMM converged : " + str(G.converged_))
    print(G.means_)

    # FV = np.zeros((epo_arr.shape[0], len(WV.keys())), dtype=float)
    FV = np.zeros((epo_arr.shape[0], n_components), dtype=float)
    i = 0
    for ess in epo_arr:  # i loop
        # print(ess)
        # j = 0
        # for w in WV.keys():   # j loop
        for j in range(n_components):
            FV[i, j] = np.inf
            ess_s = set(re.split(spl, ess))
            for wd in ess_s:
                try:
                    # print("wd: %s\tw: %s" % (wd,w))
                    # d = np.linalg.norm(WV[wd] - WV[w])
                    # print(WV[wd])
                    # print(G.means_[j, :])
                    # print(WV[wd].shape)
                    # print(G.means_[j, :].shape)
                    d = np.linalg.norm(WV[wd] - G.means_[j, :])
                    if d < FV[i, j]:
                        FV[i, j] = d
                except:
                    pass
            # j = j + 1
        i = i + 1
        if i % 10000 == 0:
            print("Iteration : %d / %d" % (i, epo_arr.shape[0]))
        # if i > 50:
        #    break

    FV = np.exp(-1.0 * FV)

    # np.save('FV', FV)
    sio.savemat('WV', {'WV_FV': FV, 'WV_Labels': labels, 'WV_tr_idx': train_idx, 'WV_te_idx': test_idx, 'WV_words': words})

    # logit_params = {'penalty': 'l2', 'dual': False, 'tol': 0.0001, 'C': 1.0, 'fit_intercept': True, 'intercept_scaling': 1, 'class_weight': None, 'random_state': None}
    linearsvc_params = {'penalty': 'l2', 'loss': 'l2', 'dual': True, 'tol': 0.0001, 'C': 1.0, 'multi_class': 'ovr', 'fit_intercept': True, 'intercept_scaling': 1, 'class_weight': None,
'verbose': 2, 'random_state': None}
    M = Learner(FV, labels)
    # P = M.fit_and_predict(train_idx, test_idx, LogisticRegression(), params=logit_params)
    P = M.fit_and_predict(train_idx, test_idx, LinearSVC(), params=linearsvc_params)

    ofile = open('word_embed_logit.csv', 'w')
    ofile.write("projectid,is_exciting\n")
    k = 0
    for i in test_idx:
            # ofile.write("%s,%f\n" % (epo.iloc[i]['projectid'], P[i][1]))
            ofile.write("%s,%f\n" % (epo.iloc[i]['projectid'], P[k]))
            k = k + 1
    ofile.close()


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
