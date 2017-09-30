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
N_GRAMS = [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5),
           (2, 2), (2, 3), (2, 4), (2, 5),
           (3, 3), (3, 4), (3, 5),
           (4, 4), (4, 5)]
MIN_FREQUENCY = [1, 2, 5]
STOP_WORDS = ["english", None]
Data = namedtuple('Data', 'data n_grams min_frequency stop_words')
TRAIN_X = []
DEV_X = []
TEST_X = []
for n_grams in N_GRAMS:
    for min_frequency in MIN_FREQUENCY:
        for stop_words in STOP_WORDS:
            vectorizer = CountVectorizer(ngram_range=n_grams, min_df=min_frequency,
                                         stop_words=stop_words)
            vectorizer.fit(TRAIN_X_RAW)
            TRAIN_X.append(Data(vectorizer.transform(TRAIN_X_RAW),
                                n_grams, min_frequency, stop_words))
            DEV_X.append(Data(vectorizer.transform(DEV_X_RAW),
                              n_grams, min_frequency, stop_words))
            TEST_X.append(Data(vectorizer.transform(TEST_X_RAW),
                               n_grams, min_frequency, stop_words))

##################################################################
# Learn LogisticRegression classifier
##################################################################
print("=======================PROBLEM 1.5=======================")
Result = namedtuple('LogisticResult', 'c train dev index')
CS = [1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1,
      1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9, 1e10]
RESULTS = []
for c in CS:
    for params_index in range(len(TRAIN_X)):
        train_x = TRAIN_X[params_index].data
        dev_x = DEV_X[params_index].data
        train_accuracy, dev_accuracy, _ = logistic_regression(train_x, TRAIN_Y, dev_x, DEV_Y, c)
        RESULTS.append(Result(c, train_accuracy, dev_accuracy, params_index))
# Print the results
print("All Results")
for res in sorted(RESULTS, key=lambda r: r.dev):
    print(res)
print("Hyperparameter Tuning Results")
BEST_RESULT = max(RESULTS, key=lambda res: res.dev)
_, TEST_ACCURACY, BEST_LR = logistic_regression(TRAIN_X[BEST_RESULT.index].data, TRAIN_Y,
                                                TEST_X[BEST_RESULT.index].data, TEST_Y, BEST_RESULT.c)
print("Best Model: " + str(BEST_RESULT))
print("Best Model Parameters: ")
print("N_GRAMS: " + str(TRAIN_X[BEST_RESULT.index].n_grams))
print("MIN_FREQUENCY: " + str(TRAIN_X[BEST_RESULT.index].min_frequency))
print("STOP_WORDS: " + str(TRAIN_X[BEST_RESULT.index].stop_words))
print("Test Accuracy of Best Model: " + str(TEST_ACCURACY))