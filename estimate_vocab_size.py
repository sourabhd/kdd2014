from __future__ import print_function
import pandas as pd
import pdb
import sys
from pprint import pprint
from nltk import wordpunct_tokenize

def main():
    essays_df = pd.read_csv('../../dataset/essays.csv')
    pprint(essays_df['essay'])
    words = set()
    count = 0
    for essay in essays_df['essay'].fillna(''):
        [words.add(w) for w in wordpunct_tokenize(essay.decode('utf-8','ignore').lower())] 
        count = count + 1
        if count % 10000 == 0:
            print(count)

    print("Vocabulary Size: %d" % len(words))
if __name__ == '__main__':
    try:
        main()
    except:
        pdb.post_mortem(sys.exc_info()[2])

