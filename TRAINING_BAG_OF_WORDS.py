# -*- coding: utf-8 -*-
"""
Function to load training data as bag of words dataframe.
"""

import pandas as pd
import nltk
import heapq
from nltk.corpus import stopwords

def getBagOfWords( data, num_freq_words):
    concatenated_tokenized_data = []
    for text in data:
        tokenized_text = nltk.word_tokenize(text)
        concatenated_tokenized_data += tokenized_text
        
    word_freq = {}
    for token in concatenated_tokenized_data:
        if token not in word_freq.keys():
            word_freq[token] = 1
        else:
            word_freq[token] += 1
            
    most_freq = heapq.nlargest(num_freq_words, word_freq, key=word_freq.get)
    sentence_vecs = []
    for text in data:
        tokens = nltk.word_tokenize(text)
        sentence_vec = []
        for word in most_freq:
            sentence_vec.append(tokens.count(word))
            
        sentence_vecs.append(sentence_vec)
    return pd.DataFrame(sentence_vecs, columns=most_freq)


def getBagOfWordsNoStop( data, num_freq_words):
    concatenated_tokenized_data = []
    for text in data:
        tokenized_text = nltk.word_tokenize(text)
        text_no_stopwords = []
        for word in tokenized_text:
            if word not in stopwords.words('english'):
                text_no_stopwords.append(word)
        concatenated_tokenized_data += text_no_stopwords

    word_freq = {}
    for token in concatenated_tokenized_data:
        if token not in word_freq.keys():
            word_freq[token] = 1
        else:
            word_freq[token] += 1
            
    most_freq = heapq.nlargest(num_freq_words, word_freq, key=word_freq.get)
    sentence_vecs = []
    for text in data:
        tokens = nltk.word_tokenize(text)
        sentence_vec = []
        for word in most_freq:
            sentence_vec.append(tokens.count(word))
            
        sentence_vecs.append(sentence_vec)
    return pd.DataFrame(sentence_vecs, columns=most_freq)
df = pd.read_csv("TRAINING_DATA.csv")
data = df["text"]
prepared_dataframe = getBagOfWordsNoStop(data=data, num_freq_words=10)
print(prepared_dataframe)
        
    