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
from nltk.tokenize import regexp

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


def getBOWasFrequencies( data):
    '''returns list of tokens as the number of times the token appears in the data set'''
    tk = regexp.WordPunctTokenizer()
    tokens = data.map(tk.tokenize)
    values = pd.Series([x for item in tokens for x in item]).value_counts()
    values = values.to_dict()
    tokens = tokens.apply(lambda x: list(map(values.get, x)))
    return tokens, values



def getBOW(text_data, num_words, has_stop, nltk_or_countVec):
    '''returns frequency of most occurent tokens in sentence. 
    
    nltk is too specific and inefficient, countVec is more efficient and 
    generic.'''
    if nltk_or_countVec == "nltk":
        concatenated_tokenized_data = []
        for text in text_data:
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
        for text in text_data:
            tokens = nltk.word_tokenize(text)
            sentence_vec = []
            for word in most_freq:
                sentence_vec.append(tokens.count(word))
                
            sentence_vecs.append(sentence_vec)
            BOW = pd.DataFrame(sentence_vecs, columns=most_freq)
        return BOW, BOW.columns 
    elif nltk_or_countVec == "countVec":
        if(has_stop):
            vectorizer = CountVectorizer(max_features=num_words)
        else:
            vectorizer = CountVectorizer(max_features=num_words, stop_words='english')
        X = vectorizer.fit_transform(text_data)
        BOW = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
        return BOW, vectorizer
    else:
        raise ValueError("must choose nltk or countVec.")

            
'''   
df = pd.read_csv("TRAINING_DATA.csv")
data = df["text"]
start_time = time.time()
print(getBOW(data, 200, False, "countVec"))
print("--- %s seconds ---" % (time.time() - start_time))
start_time = time.time()
print(getBOWasFrequencies(data))
print("--- %s seconds ---" % (time.time() - start_time))
'''