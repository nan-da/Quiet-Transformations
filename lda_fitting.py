#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 17:30:09 2017

@author: dt
"""

from glob import glob
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from tqdm import tqdm 
from time import time
from scipy import io
import pickle
import os
import random

def print_top_words(model, feature_names, n_top_words, filename=None):
    message = 'Topics in LDA model:\n'
    message += str(model.get_params())
    for topic_idx, topic in enumerate(model.components_):
        message += "\nTopic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
    if filename:
        fw = open(filename, 'w')
        fw.write(message)
        fw.close()
    print(message)

def lda_fitting(out_path,max_features,random_remove_pct=False,random_state=None, only_prominent = False):
    n_components = 150
    n_top_words = 10
    print('out path:',out_path)
    stop_words = pd.read_csv('stoplist_final.txt',header=None)[0].tolist()
    wordcount_list = glob('raw_data_quiet_trans/*/wordcounts/*.CSV') + glob('raw_data_quiet_trans/*/*/wordcounts/*.CSV')
    n_samples = len(wordcount_list)
    
    if only_prominent:
        prominent_answer = pd.read_csv('all words prominent in any topic.txt',header=None,names=['word'])
    
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    
    if random_remove_pct:
        ind = sorted(random.sample(range(n_samples), int(n_samples*(1-random_remove_pct))))
        wordcount_list = np.array(wordcount_list)[ind]
    
    def iterator_passages(only_prominent=False):
        for fn in tqdm(wordcount_list[:n_samples]):
            wcdf = pd.read_csv(fn)
            if only_prominent:
                wcdf = wcdf.merge(prominent_answer, how='inner',left_on='WORDCOUNTS',right_on='word')
                del wcdf['word']
            yield ''.join([(str(word)+' ')*count for word, count in wcdf.values])
                
    tf_vectorizer = TfidfVectorizer(max_features=max_features,
                                    min_df=2,
                                    stop_words=stop_words,
                                    use_idf=False)
    tf = tf_vectorizer.fit_transform(iterator_passages(only_prominent))
    io.mmwrite(out_path+'tf_pickle', tf) # if you want to cache
    print('tf ready')
    
    t0=time()
    lda = LatentDirichletAllocation(n_components=n_components, max_iter=5,
                                    learning_method='online',
                                    learning_offset=50.,
                                    random_state=random_state,
                                    n_jobs=1)
    lda.fit(tf)
    
    print('LDA fitting ready')
    print("done in %0.3fs." % (time() - t0))
    tf_feature_names = tf_vectorizer.get_feature_names()
    print_top_words(lda, tf_feature_names, n_top_words, filename=out_path+'lda_topics.txt')
    
    pickle.dump(lda, open(out_path+'lda_pickle.pkl', 'wb')) # if you want to cache


if __name__ == '__main__':
    lda_fitting(out_path = 'top 10k words/',
                max_features = 10000)
    
    lda_fitting(out_path = 'top 100k words/',
                max_features = 100000)

    lda_fitting(out_path = 'top 100k words random seed check/',
                max_features = 100000,
                random_state = 12345)
    
    lda_fitting(out_path = 'top 10k words random remove 1%/',
                max_features = 10000,
                random_remove_pct = 0.01)
    
    lda_fitting(out_path = 'top 100k words random remove 1%/',
                max_features = 100000,
                random_remove_pct = 0.01)