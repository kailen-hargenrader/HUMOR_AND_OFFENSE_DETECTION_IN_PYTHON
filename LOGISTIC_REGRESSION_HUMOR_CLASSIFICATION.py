# -*- coding: utf-8 -*-
"""
Use Logistic Regression to predict is_humor label 
"""
from TRAINING_BAG_OF_WORDS import getBOW
import numpy as np
import pandas as pd
import sklearn as skl
from RUN_MODELS import runClassification

df = pd.read_csv("TRAINING_DATA.csv")
training_df, test_df = skl.model_selection.train_test_split(df, test_size = .1)
training_BOW, vectorizer = getBOW(training_df["text"], 200, False, "countVec")
training_labels = training_df["is_humor"]
test_BOW = pd.DataFrame(vectorizer.transform(test_df["text"]).toarray(), columns=vectorizer.get_feature_names_out())
test_labels = test_df["is_humor"]
model = skl.linear_model.LogisticRegression()
accuracy, model = runClassification(model, training_BOW, training_labels, test_BOW, test_labels)
print(accuracy)
odds = np.exp(model.coef_[0])
odds_df = pd.DataFrame(odds, 
                    training_BOW.columns, 
                    columns=['coef']).sort_values(by='coef', ascending=False)
print(odds_df.head(10))
