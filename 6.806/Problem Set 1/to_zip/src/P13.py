from __future__ import division
from collections import namedtuple
import numpy as np
import sklearn as sk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression

##################################################################
# Helper functions
##################################################################
def get_raw_data(file_name):
    f = open(file_name)
    x_raw = []
    y = []
    for line in f:
        x_raw.append(line[2:])
        y.append(line[0])
    y = np.array(y)
    return x_raw, y

def accuracy(predicted_y, actual_y):
    return np.sum(predicted_y == actual_y)/len(actual_y)

def logistic_regression(train_x, train_y, dev_x, dev_y, c):
    lr = LogisticRegression(C=c)
    lr.fit(train_x, train_y)
    train_predicted_y = lr.predict(train_x)
    train_accurary = accuracy(train_predicted_y, train_y)
    dev_predicted_y = lr.predict(dev_x)
    dev_accurary = accuracy(dev_predicted_y, dev_y)
    return (train_accurary, dev_accurary, lr)

##################################################################
# Load and store the training, dev, and data
##################################################################
TRAIN_X_RAW, TRAIN_Y = get_raw_data("../data/stsa.binary.train")
DEV_X_RAW, DEV_Y = get_raw_data("../data/stsa.binary.dev")
TEST_X_RAW, TEST_Y = get_raw_data("../data/stsa.binary.test")
# Convert the raw training data to a bag of words representation
VECTORIZER = CountVectorizer()
VECTORIZER.fit(TRAIN_X_RAW)
TRAIN_X = VECTORIZER.transform(TRAIN_X_RAW)
DEV_X = VECTORIZER.transform(DEV_X_RAW)
TEST_X = VECTORIZER.transform(TEST_X_RAW)

##################################################################
# Learn LogisticRegression classifier
##################################################################
print("=======================PROBLEM 1.2=======================")
Result = namedtuple('LogisticResult', 'c train dev')
CS = [1e-1, 1e1, 1e3, 1e5, 1e7]
RESULTS = []
for c in CS:
    train_accuracy, dev_accuracy, _ = logistic_regression(TRAIN_X, TRAIN_Y, DEV_X, DEV_Y, c)
    RESULTS.append(Result(c, train_accuracy, dev_accuracy))
# Print the results
print("Hyperparameter Tuning Results")
BEST_P12_RESULT = max(RESULTS, key=lambda res: res.dev)
_, TEST_P12_ACCURACY, BEST_P12_LR = logistic_regression(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, BEST_P12_RESULT.c)
print("Best Model: " + str(BEST_P12_RESULT))
print("Test Accuracy of Best Model: " + str(TEST_P12_ACCURACY))

##################################################################
# Analyze Features of Best LogisticRegression Model
##################################################################
print("=======================PROBLEM 1.3=======================")
VOCABULARY = VECTORIZER.get_feature_names()
COEFS = np.abs(BEST_P12_LR.coef_[0]).tolist()
TOP_10 = sorted(zip(VOCABULARY, COEFS), key=lambda x: x[1], reverse=True)[:10]
print("Highest Impact Features")
for top in TOP_10:
    print(top)
# Analysis with stop words removed
# First, fit the vectorizer
VECTORIZER_NO_STOP_WORDS = CountVectorizer(stop_words="english")
VECTORIZER_NO_STOP_WORDS.fit(TRAIN_X_RAW)
TRAIN_X_NO_STOP_WORDS = VECTORIZER_NO_STOP_WORDS.transform(TRAIN_X_RAW)
DEV_X_NO_STOP_WORDS = VECTORIZER_NO_STOP_WORDS.transform(DEV_X_RAW)
TEST_X_NO_STOP_WORDS = VECTORIZER_NO_STOP_WORDS.transform(TEST_X_RAW)
# Now, get the best model when stop words are removed
RESULTS_NO_STOP_WORDS = []
for c in CS:
    train_accuracy, dev_accuracy, _ = logistic_regression(TRAIN_X_NO_STOP_WORDS, TRAIN_Y, DEV_X_NO_STOP_WORDS, DEV_Y, c)
    RESULTS_NO_STOP_WORDS.append(Result(c, train_accuracy, dev_accuracy))
# Print the results
print("Hyperparameter Tuning Results (for model with stop words removed)")
BEST_P13_RESULT = max(RESULTS_NO_STOP_WORDS, key=lambda res: res.dev)
_, TEST_P13_ACCURACY, BEST_P13_LR = logistic_regression(TRAIN_X_NO_STOP_WORDS, TRAIN_Y, TEST_X_NO_STOP_WORDS, TEST_Y, BEST_P13_RESULT.c)
print("Best Model: " + str(BEST_P13_RESULT))
print("Test Accuracy of Best Model: " + str(TEST_P13_ACCURACY))
# Highest Impact Features (for model with stop words removed)
VOCABULARY_NO_STOP_WORDS = VECTORIZER_NO_STOP_WORDS.get_feature_names()
COEFS_NO_STOP_WORDS = np.abs(BEST_P13_LR.coef_[0]).tolist()
TOP_10 = sorted(zip(VOCABULARY_NO_STOP_WORDS, COEFS_NO_STOP_WORDS), key=lambda x: x[1], reverse=True)[:10]
print("Highest Impact Features (for model with stop words removed)")
for top in TOP_10:
    print(top)
