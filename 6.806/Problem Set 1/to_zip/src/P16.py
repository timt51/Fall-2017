from __future__ import division
from collections import namedtuple
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

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
# Monogram model
##################################################################
print("=======================PROBLEM 1.6=======================")
DATA_PROPORTIONS = [0.005, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1]
print("======================Monogram Model======================")
ACCS = []
for prop in DATA_PROPORTIONS:
##################################################################
# Load and store the training, dev, and data
##################################################################
    TRAIN_X_RAW, TRAIN_Y = get_raw_data("../data/stsa.binary.train")
    DEV_X_RAW, DEV_Y = get_raw_data("../data/stsa.binary.dev")
    TEST_X_RAW, TEST_Y = get_raw_data("../data/stsa.binary.test")
    if prop != 1:
        TRAIN_X_RAW, _, TRAIN_Y, _ = train_test_split(TRAIN_X_RAW, TRAIN_Y, train_size=prop, random_state=55)
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
    CS = [0.1]
    RESULTS = []
    for c in CS:
        train_accuracy, dev_accuracy, _ = logistic_regression(TRAIN_X, TRAIN_Y, DEV_X, DEV_Y, c)
        RESULTS.append(Result(c, train_accuracy, dev_accuracy))
    # Print the results
    print("Hyperparameter Tuning Results For Monogram Model and prop=", prop)
    BEST_RESULT = max(RESULTS, key=lambda res: res.dev)
    _, TEST_ACCURACY, BEST_LR = logistic_regression(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, BEST_RESULT.c)
    print("Best Model: " + str(BEST_RESULT))
    print("Test Accuracy of Best Model: " + str(TEST_ACCURACY))
    ACCS.append(TEST_ACCURACY)

plt.plot(DATA_PROPORTIONS, ACCS)
plt.axis([0, 1, 0.5, 1])
plt.title("Monogram Model: Training Data Size vs Accuracy")
plt.xlabel('Proportion of Training Data')
plt.ylabel('Best Accuracy')
plt.show()

##################################################################
# Bigram model
##################################################################
print("=======================Bigram Model=======================")
ACCS = []
for prop in DATA_PROPORTIONS:
##################################################################
# Load and store the training, dev, and data
##################################################################
    TRAIN_X_RAW, TRAIN_Y = get_raw_data("../data/stsa.binary.train")
    DEV_X_RAW, DEV_Y = get_raw_data("../data/stsa.binary.dev")
    TEST_X_RAW, TEST_Y = get_raw_data("../data/stsa.binary.test")
    if prop != 1:
        TRAIN_X_RAW, _, TRAIN_Y, _ = train_test_split(TRAIN_X_RAW, TRAIN_Y, train_size=prop, random_state=42)
    # Convert the raw training data to a bag of words representation
    VECTORIZER = CountVectorizer(ngram_range=(1,2))
    VECTORIZER.fit(TRAIN_X_RAW)
    TRAIN_X = VECTORIZER.transform(TRAIN_X_RAW)
    DEV_X = VECTORIZER.transform(DEV_X_RAW)
    TEST_X = VECTORIZER.transform(TEST_X_RAW)
##################################################################
# Learn LogisticRegression classifier
##################################################################
    Result = namedtuple('LogisticResult', 'c train dev')
    CS = [10.0]
    RESULTS = []
    for c in CS:
        train_accuracy, dev_accuracy, _ = logistic_regression(TRAIN_X, TRAIN_Y, DEV_X, DEV_Y, c)
        RESULTS.append(Result(c, train_accuracy, dev_accuracy))
    # Print the results
    print("Hyperparameter Tuning Results For Bigram Model and prop=", prop)
    BEST_RESULT = max(RESULTS, key=lambda res: res.dev)
    _, TEST_ACCURACY, BEST_LR = logistic_regression(TRAIN_X, TRAIN_Y, TEST_X, TEST_Y, BEST_RESULT.c)
    print("Best Model: " + str(BEST_RESULT))
    print("Test Accuracy of Best Model: " + str(TEST_ACCURACY))
    ACCS.append(TEST_ACCURACY)

plt.plot(DATA_PROPORTIONS, ACCS)
plt.axis([0, 1, 0.5, 1])
plt.title("Bigram Model: Training Data Size vs Accuracy")
plt.xlabel('Proportion of Training Data')
plt.ylabel('Best Accuracy')
plt.show()
