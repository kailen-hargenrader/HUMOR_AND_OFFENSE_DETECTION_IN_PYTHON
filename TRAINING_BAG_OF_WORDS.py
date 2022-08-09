# -*- coding: utf-8 -*-
"""
Function to load training data as bag of words dataframe.
"""

import pandas as pd
import nltk
import heapq
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import time

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

def getBagOfWordsCountVec( data, num_freq_words):
    vectorizer = CountVectorizer(max_features=num_freq_words)
    X = vectorizer.fit_transform(data)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), )

def getBagOfWordsCountVecNoStop( data, num_freq_words):
    vectorizer = CountVectorizer(max_features=num_freq_words, stop_words='english')
    X = vectorizer.fit_transform(data)
    return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), )

#nltk is too specific: Case sensitive,  countVec is more generic
def GetBOW(text_data, num_words, has_stop, nltk_or_countVec):
    if nltk_or_countVec == "nltk":
        concatenated_tokenized_data = []
        for text in data:
            tokenized_text = nltk.word_tokenize(text)
            if(has_stop):
                concatenated_tokenized_data +=tokenized_text
            else:
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
                
        most_freq = heapq.nlargest(num_words, word_freq, key=word_freq.get)
        sentence_vecs = []
        for text in data:
            tokens = nltk.word_tokenize(text)
            sentence_vec = []
            for word in most_freq:
                sentence_vec.append(tokens.count(word))
                
            sentence_vecs.append(sentence_vec)
        return pd.DataFrame(sentence_vecs, columns=most_freq)
    elif nltk_or_countVec == "countVec":
        if(has_stop):
            vectorizer = CountVectorizer(max_features=num_words)
        else:
            vectorizer = CountVectorizer(max_features=num_words, stop_words='english')
        X = vectorizer.fit_transform(data)
        return pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out(), )
    else:
        raise ValueError("must choose nltk or countVec.")

df = pd.read_csv("TRAINING_DATA.csv")
data = df["text"]
getBagOfWords(data, 10)
'''
start_time = time.time()
print(GetBOW(data, 10, False, "countVec"))
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print(GetBOW(data, 10, False, "nltk"))
print("--- %s seconds ---" % (time.time() - start_time))
'''