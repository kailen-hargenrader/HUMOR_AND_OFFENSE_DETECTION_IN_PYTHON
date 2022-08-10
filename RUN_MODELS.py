# -*- coding: utf-8 -*-
"""
Run various models on training data using validation data for accuracy.
The purpose is to find the most promising model type
"""

import TRAINING_BAG_OF_WORDS as BOW
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn as skl

def runClassification(model, training_BOW, training_labels, test_BOW, test_labels):
    #print("0----------------------100")
    #print(" ", end = "")
        model.fit(training_BOW, training_labels)
        predict_labels = model.predict(test_BOW)
        accuracy_score = skl.metrics.accuracy_score(test_labels, predict_labels)
        return accuracy_score
    


df = pd.read_csv("TRAINING_DATA.csv")
training_df, test_df = skl.model_selection.train_test_split(df, test_size = .1)
training_BOW, vectorizer = BOW.getBOW(training_df["text"], 200, False, "countVec")
training_labels = training_df["is_humor"]
test_BOW = pd.DataFrame(vectorizer.transform(test_df["text"]).toarray(), columns=vectorizer.get_feature_names_out())
test_labels = test_df["is_humor"]
model = skl.linear_model.LogisticRegression()
accuracy = runClassification(model, training_BOW, training_labels, test_BOW, test_labels)
print(accuracy)
    