# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 07:56:48 2019

@author: bmccs
"""

# New commentary fix construction pipeline

import sqlite3
import numpy as np
import pandas as pd
import pickle

conn=sqlite3.connect('bible.db')
results=conn.cursor().execute("SELECT book,chapter_verse,content,cross_references FROM T_Bible").fetchall()
keys=list(map(lambda x: " ".join([x[0], x[1]]), results) )
keys_content_refs=list(map(lambda x: (" ".join([x[0], x[1]]),x[2],x[3]), results) )

int2verse={}
verse2int={}
for i,v in enumerate(keys):
    int2verse[i]=v
    verse2int[v]=i

print("int2verse")
print(int2verse)
print("verse2int")
print(verse2int)

kjv_bible_mapping={}
num_verses=len(int2verse.keys())
i=0
for k,content,refs in keys_content_refs:
    #print(k, obj[0], obj[1])
    mapping=np.zeros((num_verses))
    for cf in refs.split(","):
        mapping[verse2int[cf]]=1
    kjv_bible_mapping[k]=[content,mapping] #dataset [X|y]
    i+=1
    
    if i % 1000 == 0:
        print(i)

print("Genesis 1:1 mapping")
print(kjv_bible_mapping["Genesis 1:1"][1])

"""
print("Saving")
with open('kjv_bible_mapping.pkl', 'wb') as handle:
    pickle.dump(kjv_bible_mapping, handle)
with open('int2verse_mapping.pkl', 'wb') as handle:
    pickle.dump(int2verse, handle)
with open('verse2int_mapping.pkl', 'wb') as handle:
    pickle.dump(verse2int, handle)
print("Successful saving.")

print("Loading to test stability.")
with open('kjv_bible_mapping.pkl', 'rb') as handle:
    kjv_bible_mapping=pickle.load(handle)
with open('int2verse_mapping.pkl', 'rb') as handle:
    int2verse=pickle.load(handle)
with open('verse2int_mapping.pkl', 'rb') as handle:
    verse2int=pickle.load(handle)
print("Finished loading.")
"""


# In[4]:


import nltk
import string
from nltk import word_tokenize          
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
#nltk.download('punkt')

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

def tokenize(text, stemmer = PorterStemmer()):
    text = "".join([ch for ch in text if ch not in string.punctuation])
    tokens = nltk.word_tokenize(text)
    stems = stem_tokens(tokens, stemmer)
    return stems

print("Building corpus")
corpus=list(map(lambda x:x[1][0], kjv_bible_mapping.items()))
print(corpus[:5])
print("Finished")

# In[5]:


print("Building tfidf")
tfidf_vectorizer = TfidfVectorizer(#tokenizer=tokenize, 
                                   stop_words="english",
                                   lowercase=True,
                                   norm='l2')
tfidf_fit=tfidf_vectorizer.fit(corpus)
tfidf_matrix = tfidf_fit.transform(corpus)
print("Finished")
# ## Parallelize the transformation
# <hr>
# Freaky fast..

# In[13]:

"""
import multiprocessing

from multiprocessing import Pool
import scipy.sparse as sp
#num_partitions = 5
num_cores = multiprocessing.cpu_count()
num_partitions = num_cores-1 # I like to leave some cores for other
#processes
print(num_partitions)

def parallelize_dataframe(df, func):
    a = np.array_split(df, num_partitions)
    del df
    pool = Pool(num_cores)
    #df = pd.concat(pool.map(func, [a,b,c,d,e]))
    df = sp.vstack(pool.map(func, a), format='csr')
    pool.close()
    pool.join()
    return df

def test_func(data):
    #print("Process working on: ",data)
    tfidf_matrix = tfidf_fit.transform(data)
    #return pd.DataFrame(tfidf_matrix.toarray())
    return tfidf_matrix
"""


print("Saving out tfidf fit and mat.")
#tfidf_parallel = parallelize_dataframe(corpus, test_func)
with open("tfidf_bible_matrix.pkl", "wb") as handle:
    pickle.dump(tfidf_matrix,handle)
with open("tfidf_bible_fit.pkl", "wb") as handle:
    pickle.dump(tfidf_fit,handle)
with open("tfidf_bible_matrix.pkl", "rb") as handle:
    tf_idf_bible_matrix=pickle.load(handle)
with open("tfidf_bible_fit.pkl", "rb") as handle:
    tf_idf_bible_fit=pickle.load(handle)
print("Finished")

#pickle_dump(tfidf_parallel, 'tfidf_bible_matrix.pkl')
#pickle_dump(tfidf_fit, 'tfidf_bible_fit.pkl')
#tf_idf_bible_matrix=pickle_load('tfidf_bible_matrix.pkl')
#tf_idf_bible_fit=pickle_load('tfidf_bible_fit.pkl')
#print(tf_idf_bible_matrix[:4])


# ## Construct the model

# In[7]:


import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Flatten
from keras.callbacks import ModelCheckpoint
from keras.utils import np_utils

X=tf_idf_bible_matrix.todense()
y=np.array(list(map(lambda x:x[1][1], kjv_bible_mapping.items())))

print(X.shape)
print(y.shape)
print(len(corpus))
print(len(list(kjv_bible_mapping.keys())))


# In[8]:


from keras.layers import Dense, Dropout, Conv1D, MaxPooling1D
from keras import metrics
import os
import datetime
def create_model(X,y):
    # Input layers
    print(X.shape)
    print(y.shape)
    # (31102, 12302)
    # 
    model = Sequential()
    model.add(Dense(10000, input_shape=(X.shape[1],)))
    model.add(Dense(1000))
    model.add(Dense(100))
    model.add(Dense(y.shape[1], activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['categorical_accuracy'])
    return model
model_path="weights-improvement-01-54.2576.hdf5"
def load_trained_model(weights_path, X, y):
    model = create_model(X,y)
    model.load_weights(weights_path)
    print("Loaded")
    return model

#model=load_trained_model(model_path,X,y)



# ## Train step
# Be careful before you do this, may take 20 mins.

# In[ ]:


filepath="../data/weights-improvement-{epoch:02d}-{loss:.4f}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]

rnn_model = create_model(X,y)
rnn_model.summary()

rnn_model.fit(X, y,
              batch_size=128,
              epochs=1,
              callbacks=callbacks_list)

