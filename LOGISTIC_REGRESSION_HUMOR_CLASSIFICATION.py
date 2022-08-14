# -*- coding: utf-8 -*-
"""
Use Logistic Regression to predict is_humor label 
"""
from TRAINING_BAG_OF_WORDS import getBOW
import numpy as np
import pandas as pd
import sklearn as skl
import matplotlib.pyplot as plt
from RUN_MODELS import runClassification

df = pd.read_csv("TRAINING_DATA.csv")
training_df, test_df = skl.model_selection.train_test_split(df, test_size = .1)
training_BOW, vectorizer = getBOW(training_df["text"], 200, False, "countVec")
training_labels = training_df["is_humor"]
test_BOW = pd.DataFrame(vectorizer.transform(test_df["text"]).toarray(),
                        columns=vectorizer.get_feature_names_out())
test_labels = test_df["is_humor"]

accuracy_arr = []
for i in range(1, 100, 10):
    model = skl.linear_model.LogisticRegressionCV(Cs=i)
    accuracy, model = runClassification(model, training_BOW, 
                                        training_labels, test_BOW, test_labels)
    accuracy_arr.append(accuracy)
    odds = np.exp(model.coef_[0])
    odds_df = pd.DataFrame(odds, 
                        training_BOW.columns, 
                        columns=['coef']).sort_values(by='coef', 
                                                      ascending=False)
complexity_penalty_arr = np.arange(1, 100, 10)                                                    
plt.plot(complexity_penalty_arr, accuracy_arr, label='Cross Validation', color='red')
plt.axhline(y=np.mean(accuracy_arr), color='r',label='Cross Validation Mean', linestyle='--')

accuracy_arr = []
for i in range(1, 100, 10):
    model = skl.linear_model.LogisticRegression(C=i)
    accuracy, model = runClassification(model, training_BOW, 
                                        training_labels, test_BOW, test_labels)
    accuracy_arr.append(accuracy)
    odds = np.exp(model.coef_[0])
    odds_df = pd.DataFrame(odds, 
                        training_BOW.columns, 
                        columns=['coef']).sort_values(by='coef', 
                                                      ascending=False)
                                                      
plt.plot(complexity_penalty_arr, accuracy_arr, label='Holdout', color='blue')
plt.axhline(y=np.mean(accuracy_arr),label='Holdout Mean', color='blue', linestyle='--')  
plt.legend(loc='lower right')