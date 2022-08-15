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

def compareCV(min_complexity_penalty, max_complexity_penalty, step):
    accuracy_arr = []
    for i in range(min_complexity_penalty, max_complexity_penalty, step):
        CV_model = skl.linear_model.LogisticRegressionCV(Cs=i)
        accuracy, CV_model = runClassification(CV_model, training_BOW, 
                                            training_labels, test_BOW, test_labels)
        accuracy_arr.append(accuracy)
        
    complexity_penalty_arr = np.arange(min_complexity_penalty, max_complexity_penalty, step)                                                    
    plt.plot(complexity_penalty_arr, accuracy_arr, label='Cross Validation', color='red')
    plt.axhline(y=np.mean(accuracy_arr), color='r',label='Cross Validation Mean', linestyle='--')
    
    accuracy_arr = []
    for i in range(min_complexity_penalty, max_complexity_penalty, step):
        model = skl.linear_model.LogisticRegression(C=i)
        accuracy, model = runClassification(model, training_BOW, 
                                            training_labels, test_BOW, test_labels)
        accuracy_arr.append(accuracy)
        
                                                          
    plt.plot(complexity_penalty_arr, accuracy_arr, label='Holdout', color='blue')
    plt.axhline(y=np.mean(accuracy_arr),label='Holdout Mean', color='blue', linestyle='--')  
    plt.legend(loc='lower right')
    return CV_model, model

def compareCoef(base_model, comparison_model):
    comparison_odds = np.exp(comparison_model.coef_[0])
    comparison_df = pd.DataFrame(comparison_odds, 
                        training_BOW.columns, 
                        columns=['CV_Coef'])
    odds = np.exp(base_model.coef_[0])
    comparison_df['Base_Coef'] = odds
    comparison_df['Difference'] = comparison_df['CV_Coef'] - comparison_df['Base_Coef']
    comparison_df = comparison_df.reindex(comparison_df.Difference.abs().sort_values(ascending=False).index)
    return comparison_df
    
df = pd.read_csv("TRAINING_DATA.csv")
training_df, test_df = skl.model_selection.train_test_split(df, test_size = .1)
training_BOW, vectorizer = getBOW(training_df["text"], 200, False, "countVec")
training_labels = training_df["is_humor"]
test_BOW = pd.DataFrame(vectorizer.transform(test_df["text"]).toarray(),
                        columns=vectorizer.get_feature_names_out())
test_labels = test_df["is_humor"]
base, CV = compareCV(1,20,2)
comparison = compareCoef(base, CV)
print(comparison.head(20))