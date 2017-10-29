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

# Create parser for input arguments
PARSER = argparse.ArgumentParser(description='Gene tagging with maximum entropy model')
PARSER.add_argument('training_set_path', help='path of training set')
PARSER.add_argument('test_set_path', help='path of test set')
PARSER.add_argument('model_type', type=int, help='type of model (1 or 2 or 3)')

# Parse arguments and display the parsed arguments
print('Beginning gene tag training and evaluation...')
ARGS = PARSER.parse_args()
print('Using training set at ' + ARGS.training_set_path)
print('Using test set at ' + ARGS.test_set_path)
print('Using model type ' + str(ARGS.model_type))
print('Assuming dev set located at data/dev.tag')
ARGS.dev_set_path = 'data/dev.tag'

# Parse data sets into word and tag sequences
TRAIN_IDENTIFIERS, TRAIN_X_RAW, TRAIN_Y = helpers.parse_data(ARGS.training_set_path)
DEV_IDENTIFIERS, DEV_X_RAW, DEV_Y = helpers.parse_data(ARGS.dev_set_path)
TEST_IDENTIFIERS, TEST_X_RAW, TEST_Y = helpers.parse_data(ARGS.test_set_path)
print('Proportion of GENE in training set', sum(TRAIN_Y)/len(TRAIN_Y))
print('Proportion of GENE in dev set', sum(DEV_Y)/len(DEV_Y))

# Convert data sets to feature representations, train model(s),
# and evaluate model(s)
# TODO: word_vectorizer - stopwords?
# TODO: char_vectorizer - or char_wb?, ngram_range, 
# TODO: logistyic Cs
if ARGS.model_type == 1:
    # parameters for context free model
    N_WORDS = [0] #[0, 1, 2, 3]
    N_CHARS = [0] #[0, 1, 2, 3]
    N_TAGS = [0] #[0, 1, 2, 3]
    NGRAMS_MAX = [4] # [1, 2, 3, 4, 5]
elif (ARGS.model_type == 2 or ARGS.model_type == 3):
    # parameters for context sensitive models
    N_WORDS = [0] #[0, 1, 2, 3]
    N_CHARS = [0] #[0, 1, 2, 3]
    N_TAGS = [0] #[0, 1, 2, 3]
    NGRAMS_MAX = [4] # [1, 2, 3, 4, 5]
elif ARGS.model_type == 4:
    # parameters for context free hyperparameter tuning
    # TODO: fix
    N_WORDS = [0]
    N_CHARS = [0]
    N_TAGS = [0]
    NGRAMS_MAX = [1, 2, 3, 4, 5, 6] #[4]
else:
    # parameters for context sensitive hyperparameter tuning
    # TODO: fix
    N_WORDS = [0,1,2,3] #[1, 2, 3]
    N_CHARS = [0,1,2,3] #[1, 2, 3]
    N_TAGS = [1,2,3] #[1, 2, 3]
    NGRAMS_MAX = [1,2,3,4,5] #[1, 2, 3, 4, 5]
TO_CACHE = True
Result = namedtuple('Result', 'n_words n_chars n_tags ngram_max dev_f1')
RESULTS = []
for n_words, n_chars, n_tags, ngram_max in itertools.product(N_WORDS, N_CHARS, N_TAGS, NGRAMS_MAX):
    # Generate feature vectors for training data
    print('Training with n_words=' + str(n_words) + ' n_chars=' + str(n_chars) + ' n_tags=' + str(n_tags) + ' ngram_max=' + str(ngram_max))
    print('Generating feature vectors for training set...')
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
    print('Training...')
    lr = LogisticRegression()
    lr.fit(train_x, TRAIN_Y)
    train_predicted_y = lr.predict(train_x)
    train_accurary = helpers.accuracy(train_predicted_y, TRAIN_Y)
    print('Train accuracy: ' + str(train_accurary))
    word_feature_names = word_vectorizer.get_feature_names()
    char_feature_names = char_vectorizer.get_feature_names()
    coefs = np.abs(lr.coef_[0]).tolist()
    top10 = sorted(zip(word_feature_names*n_words+char_feature_names, coefs), key=lambda x: x[1], reverse=True)[:10]
    print("Highest Impact Features")
    for top in top10:
        print(top)

    # Evaluate on the dev set
    print('Evaluating on dev set...')
    if ARGS.model_type == 1 or ARGS.model_type == 4:
        print('Generating features for dev set...')
        if TO_CACHE is True:
            cache = 'dev' + '_n_words_' + str(n_words) + '_n_chars_' + str(n_chars) + '_n_tags_' + str(n_tags) + '_ngrams_max_' + str(ngram_max)
        else:
            cache = None
        dev_x = helpers.extractMaxEntFeatures(DEV_X_RAW, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples, cache, predicted_ys=DEV_Y)
        dev_predicted_y = lr.predict(dev_x)
        dev_accuracy = helpers.accuracy(dev_predicted_y, DEV_Y)
        print('Dev accuracy: ' + str(dev_accuracy))
        dev_f1_score = helpers.evaluateLogisticRegressionModel(DEV_X_RAW, DEV_IDENTIFIERS, dev_predicted_y)
        RESULTS.append(Result(n_words, n_chars, n_tags, ngram_max, dev_f1_score))

    elif ARGS.model_type == 2 or ARGS.model_type == 5:
        print('Generating features for dev set...')
        word_vocabulary_size = len(word_vectorizer.vocabulary_)
        char_vocabulary_size = len(char_vectorizer.vocabulary_)
        num_features = (n_words + 1) * word_vocabulary_size + \
                        1 * char_vocabulary_size + \
                        3 * n_tags
        print('Generating dev set predictions...')
        predicted_ys = []
        for word_sequence in DEV_X_RAW:
            word_sequence = word_sequence.split(' ')
            predicted_sequence_ys = []
            for word_index in range(len(word_sequence)):
                x = helpers.featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features, predicted_ys=predicted_sequence_ys)
                predicted_y = lr.predict(x)
                predicted_sequence_ys.append(predicted_y)
            predicted_ys.extend(predicted_sequence_ys)
        
        print('Generating dev set evaluations...')
        if TO_CACHE is True:
            cache = 'dev'
        else:
            cache = None
        dev_predicted_y = np.array(predicted_ys).T[0]
        dev_accuracy = helpers.accuracy(dev_predicted_y, DEV_Y)
        print('Dev accuracy: ' + str(dev_accuracy))
        dev_f1_score = helpers.evaluateLogisticRegressionModel(DEV_X_RAW, DEV_IDENTIFIERS, dev_predicted_y)
        RESULTS.append(Result(n_words, n_chars, n_tags, ngram_max, dev_f1_score))

    elif ARGS.model_type == 3 or ARGS.model_type == 6:
        print('Generating features for dev set...')
        word_vocabulary_size = len(word_vectorizer.vocabulary_)
        char_vocabulary_size = len(char_vectorizer.vocabulary_)
        num_features = (n_words + 1) * word_vocabulary_size + \
                        1 * char_vocabulary_size + \
                        3 * n_tags
        print('Generating dev set predictions...')
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
                    # print('word_index: ', word_index, 'curr_tag_hist: ', prev_tag_seqs[curr_tag_hist_i], \
                    #         'predicted_ys: ', [0 for _ in range(word_index-n_tags)]+prev_tag_seqs[curr_tag_hist_i])
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
            # print('sequence:', word_sequence)
            # print('subproblem matrix...')
            # for row in subproblems:
            #     print(row*100)
            # print('optimal sequences...')
            # for k, v in optimal_sequences.items():
            #     print(k, v)
            # Now using subproblems... get prediction...
            best_tag_end_index = np.argmax(subproblems[:, -1])
            if (best_tag_end_index, len(word_sequence)-1) in optimal_sequences:
                predicted_sequence_ys = optimal_sequences[(best_tag_end_index, len(word_sequence)-1)]
            else:
                predicted_sequence_ys = [0 for _ in range(len(word_sequence))]
            predicted_ys.extend(predicted_sequence_ys)
        
        print(len(predicted_ys))
        print(len(DEV_Y))
        print('Generating dev set evaluations...')
        dev_predicted_y = np.array(predicted_ys)
        dev_accuracy = helpers.accuracy(dev_predicted_y, DEV_Y)
        print('Dev accuracy: ' + str(dev_accuracy))
        dev_f1_score = helpers.evaluateLogisticRegressionModel(DEV_X_RAW, DEV_IDENTIFIERS, dev_predicted_y)
        RESULTS.append(Result(n_words, n_chars, n_tags, ngram_max, dev_f1_score))

    # Predict on the test set

if ARGS.model_type > 3:
    print('Results sorted by F1 score on dev set...')
    RESULTS = sorted(RESULTS, key=lambda x: x.dev_f1, reverse=True)
    for result in RESULTS:
        print(result)