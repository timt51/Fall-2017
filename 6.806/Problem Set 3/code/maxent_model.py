from __future__ import division
import subprocess
import argparse
import re
import helpers
import itertools
from functools import reduce
from itertools import product
from operator import add
from collections import namedtuple
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import time

# Create parser for input arguments
PARSER = argparse.ArgumentParser(description='Gene tagging with maximum entropy model')
PARSER.add_argument('training_set_path', help='path of training set')
PARSER.add_argument('test_set_path', help='path of test set')
PARSER.add_argument('model_type', type=int, help='type of model (1 or 2 or 3)')

# Parse arguments and display the parsed arguments
ARGS = PARSER.parse_args()
ARGS.dev_set_path = 'data/dev.tag'

# Parse data sets into word and tag sequences
TRAIN_IDENTIFIERS, TRAIN_X_RAW, TRAIN_Y = helpers.parse_data(ARGS.training_set_path)
DEV_IDENTIFIERS, DEV_X_RAW, DEV_Y = helpers.parse_data(ARGS.dev_set_path)
TEST_IDENTIFIERS, TEST_X_RAW, TEST_Y = helpers.parse_data(ARGS.test_set_path)
DEV_IDENTIFIERS, DEV_X_RAW, DEV_Y = TEST_IDENTIFIERS, TEST_X_RAW, TEST_Y

# Convert data sets to feature representations, train model(s),
# and evaluate model(s)
if ARGS.model_type == 1:
    # parameters for context free model
    N_WORDS = [0] #[0, 1, 2, 3]
    N_CHARS = [0] #[0, 1, 2, 3]
    N_TAGS = [0] #[0, 1, 2, 3]
    NGRAMS_MAX = [4] # [1, 2, 3, 4, 5]
elif (ARGS.model_type == 2):
    # parameters for context sensitive models
    N_WORDS = [2] #[0, 1, 2, 3]
    N_CHARS = [0] #[0, 1, 2, 3]
    N_TAGS = [1] #[0, 1, 2, 3]
    NGRAMS_MAX = [4] # [1, 2, 3, 4, 5]
elif (ARGS.model_type == 3):
    # parameters for context sensitive models
    N_WORDS = [3] #[0, 1, 2, 3]
    N_CHARS = [0] #[0, 1, 2, 3]
    N_TAGS = [1] #[0, 1, 2, 3]
    NGRAMS_MAX = [5] # [1, 2, 3, 4, 5]
elif ARGS.model_type == 4:
    # parameters for context free hyperparameter tuning
    # TODO: fix
    N_WORDS = [0]
    N_CHARS = [0]
    N_TAGS = [0]
    NGRAMS_MAX = [1, 2, 3, 4, 5, 6]
else:
    # parameters for context sensitive hyperparameter tuning
    # TODO: fix
    N_WORDS = [2]
    N_CHARS = [0]
    N_TAGS = [1]
    NGRAMS_MAX = [4]
TO_CACHE = None
Result = namedtuple('Result', 'n_words n_chars n_tags ngram_max dev_p dev_r dev_f1')
RESULTS = []
for n_words, n_chars, n_tags, ngram_max in (itertools.product(N_WORDS, N_CHARS, N_TAGS, NGRAMS_MAX)):
    st = time.time()
    # Generate feature vectors for training data
    word_vectorizer = CountVectorizer(analyzer='word', binary=True)
    word_vectorizer.fit(TRAIN_X_RAW)
    char_vectorizer = CountVectorizer(analyzer='char', binary=True, lowercase=False, ngram_range=(1, ngram_max))
    char_vectorizer.fit(TRAIN_X_RAW)
    num_samples = len(TRAIN_Y)
    if TO_CACHE is True:
        cache = 'train' + '_n_words_' + str(n_words) + '_n_chars_' + str(n_chars) + '_n_tags_' + str(n_tags) + '_ngrams_max_' + str(ngram_max)
    else:
        cache = None
    train_x = helpers.extractMaxEntFeatures(TRAIN_X_RAW, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples, cache=cache, predicted_ys=TRAIN_Y)

    # Train the logistic model
    lr = LogisticRegression()
    lr.fit(train_x, TRAIN_Y)
    train_predicted_y = lr.predict(train_x)
    train_accurary = helpers.accuracy(train_predicted_y, TRAIN_Y)
    word_feature_names = word_vectorizer.get_feature_names()
    char_feature_names = char_vectorizer.get_feature_names()
    coefs = np.abs(lr.coef_[0]).tolist()
    top10 = sorted(zip(word_feature_names*n_words+char_feature_names, coefs), key=lambda x: x[1], reverse=True)[:10]

    # Evaluate on the dev set
    if ARGS.model_type == 1 or ARGS.model_type == 4:
        if TO_CACHE is True:
            cache = 'dev' + '_n_words_' + str(n_words) + '_n_chars_' + str(n_chars) + '_n_tags_' + str(n_tags) + '_ngrams_max_' + str(ngram_max)
        else:
            cache = None
        dev_x = helpers.extractMaxEntFeatures(DEV_X_RAW, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples, cache, predicted_ys=DEV_Y)
        dev_predicted_y = lr.predict(dev_x)
        dev_accuracy = helpers.accuracy(dev_predicted_y, DEV_Y)
        helpers.evaluateLogisticRegressionModelPrint(DEV_X_RAW, DEV_IDENTIFIERS, dev_predicted_y)

    elif ARGS.model_type == 2 or ARGS.model_type == 5:
        word_vocabulary_size = len(word_vectorizer.vocabulary_)
        char_vocabulary_size = len(char_vectorizer.vocabulary_)
        num_features = (n_words + 1) * word_vocabulary_size + \
                        1 * char_vocabulary_size + \
                        3 * n_tags
        predicted_ys = []
        for word_sequence in DEV_X_RAW:
            word_sequence = word_sequence.split(' ')
            predicted_sequence_ys = []
            for word_index in range(len(word_sequence)):
                x = helpers.featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features, predicted_ys=predicted_sequence_ys)
                predicted_y = lr.predict(x)
                predicted_sequence_ys.append(predicted_y)
            predicted_ys.extend(predicted_sequence_ys)
        
        if TO_CACHE is True:
            cache = 'dev'
        else:
            cache = None
        dev_predicted_y = np.array(predicted_ys).T[0]
        dev_accuracy = helpers.accuracy(dev_predicted_y, DEV_Y)
        helpers.evaluateLogisticRegressionModelPrint(DEV_X_RAW, DEV_IDENTIFIERS, dev_predicted_y)

    elif ARGS.model_type == 3 or ARGS.model_type == 6:
        word_vocabulary_size = len(word_vectorizer.vocabulary_)
        char_vocabulary_size = len(char_vectorizer.vocabulary_)
        num_features = (n_words + 1) * word_vocabulary_size + \
                        1 * char_vocabulary_size + \
                        3 * n_tags
        prev_tag_seqs = dict(zip(range(2**n_tags), product(*((0,1) for _ in range(n_tags)))))
        prev_tag_seqs_inv = {v: k for k, v in prev_tag_seqs.items()}
        prev_tag_seqs = dict((k, list(v)) for k,v in prev_tag_seqs.items())
        predicted_ys = []
        for word_sequence in DEV_X_RAW:
            word_sequence = word_sequence.split(' ')
            subproblems = np.zeros((2**n_tags, len(word_sequence)))
            optimal_sequences = {}
            for word_index in range(0, len(word_sequence)):
                curr_log_probas = {}
                for curr_tag_hist_i in range(2**n_tags):
                    x = helpers.featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, \
                                            n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, \
                                            num_features, \
                                            predicted_ys=[0 for _ in range(word_index-n_tags)]+prev_tag_seqs[curr_tag_hist_i])
                    predicted_log_proba = lr.predict_log_proba(x)[0]
                    curr_log_probas[(0,) + tuple(prev_tag_seqs[curr_tag_hist_i])] = predicted_log_proba[0]
                    curr_log_probas[(1,) + tuple(prev_tag_seqs[curr_tag_hist_i])] = predicted_log_proba[1]
                for curr_tag_hist_i in range(2**n_tags):
                    tag_hist = prev_tag_seqs[curr_tag_hist_i]
                    prev_tag_hist_i = prev_tag_seqs_inv[(0,)+tuple(tag_hist[:-1])]
                    prev_gene_hist_i = prev_tag_seqs_inv[(1,)+tuple(tag_hist[:-1])]
                    tag_log_p = subproblems[prev_tag_hist_i, word_index-1] + curr_log_probas[(tag_hist[-1],) + (0,)+tuple(tag_hist[:-1])]
                    gene_log_p = subproblems[prev_gene_hist_i, word_index-1] + curr_log_probas[(tag_hist[-1],) + (1,)+tuple(tag_hist[:-1])]
                    if gene_log_p >= tag_log_p:
                        if (prev_gene_hist_i, word_index-1) not in optimal_sequences:
                            optimal_sequences[(prev_gene_hist_i, word_index-1)] = []
                        optimal_sequences[(curr_tag_hist_i, word_index)] = optimal_sequences[(prev_gene_hist_i, word_index-1)] + [tag_hist[-1]]
                        subproblems[curr_tag_hist_i, word_index] = gene_log_p
                    else:
                        if (prev_tag_hist_i, word_index-1) not in optimal_sequences:
                            optimal_sequences[(prev_tag_hist_i, word_index-1)] = []
                        optimal_sequences[(curr_tag_hist_i, word_index)] = optimal_sequences[(prev_tag_hist_i, word_index-1)] + [tag_hist[-1]]
                        subproblems[curr_tag_hist_i, word_index] = tag_log_p
            # Now using subproblems... get prediction...
            best_tag_end_index = np.argmax(subproblems[:, -1])
            if (best_tag_end_index, len(word_sequence)-1) in optimal_sequences:
                predicted_sequence_ys = optimal_sequences[(best_tag_end_index, len(word_sequence)-1)]
            else:
                predicted_sequence_ys = [0 for _ in range(len(word_sequence))]
            predicted_ys.extend(predicted_sequence_ys)
        
        dev_predicted_y = np.array(predicted_ys)
        dev_accuracy = helpers.accuracy(dev_predicted_y, DEV_Y)
        helpers.evaluateLogisticRegressionModelPrint(DEV_X_RAW, DEV_IDENTIFIERS, dev_predicted_y)

    # Predict on the test set