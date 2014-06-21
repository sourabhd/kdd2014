from __future__ import print_function
import pandas as pd
import numpy as np
import pdb
import sys
import re
import string
from time import time


def tolower(s):
        try:
            return " ".join(re.findall(r'\w+', s,flags = re.UNICODE | re.LOCALE)).lower().decode('utf-8')
        except:
            return " ".join(re.findall(r'\w+', "no_text",flags = re.UNICODE | re.LOCALE)).lower().decode('utf-8')

def removepunct(s):
    try:
        return ''.join([ word.encode('utf-8').translate(None,string.punctuation) for word in s ])
    except:
        return ''

def main():
#   Read all data
    essays = pd.read_csv(open('../../dataset/essays.csv','r'))

#    Cleanup essays text data 
    essays['essay_cl'] = essays['essay'].apply(tolower).apply(removepunct)
    ess_proj_arr = np.array(essays['essay_cl'],dtype='string')
    f = open('essay_corpus.txt','w')
    for x in ess_proj_arr:
        f.write(str(x) + ' ')
    f.close()

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

