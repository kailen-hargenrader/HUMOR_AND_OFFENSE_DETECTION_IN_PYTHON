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

def runClassification(model, training_BOW, 
                      training_labels, test_BOW, test_labels):
    #print("0----------------------100")
    #print(" ", end = "")
        model.fit(training_BOW, training_labels)
        predict_labels = model.predict(test_BOW)
        accuracy_score = skl.metrics.accuracy_score(test_labels, 
                                                    predict_labels)
        return accuracy_score, model
    
    