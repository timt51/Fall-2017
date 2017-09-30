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
    return (train_accurary, dev_accurary)

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
Result = namedtuple('LogisticResult', 'c train dev')
CS = [1e-1, 1e1, 1e3, 1e5, 1e7]
RESULTS = []
for c in CS:
    train_accuracy, dev_accuracy = logistic_regression(TRAIN_X, TRAIN_Y, DEV_X, DEV_Y, c)
    RESULTS.append(Result(c, train_accuracy, dev_accuracy))
# Print the results
print("Hyperparameter Tuning Results")
BEST_RESULT = max(RESULTS, key=lambda res: res.dev)
_, TEST_ACCURACY = logistic_regression(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, BEST_RESULT.c)
print("Best Model: " + str(BEST_RESULT))
print("Test Accuracy of Best Model: " + str(TEST_ACCURACY))