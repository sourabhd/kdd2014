make
time ./word2vec -train essay_corpus.txt -output vectors.txt -cbow 0 -size 200 -window 5 -negative 0 -hs 1 -sample 1e-3 -threads 12 -binary 0
