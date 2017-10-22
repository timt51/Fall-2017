from __future__ import division
import argparse
import helpers
import itertools
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

PARSER = argparse.ArgumentParser(description='Gene tagging with maximum entropy model')
PARSER.add_argument('training_set_path', help='path of training set')
PARSER.add_argument('test_set_path', help='path of test set')
PARSER.add_argument('model_type', type=int, help='type of model (1 or 2 or 3)')

print('Beginning gene tag training and evaluation...')
ARGS = PARSER.parse_args()
print('Using training set at ' + ARGS.training_set_path)
print('Using test set at ' + ARGS.test_set_path)
print('Using model type ' + str(ARGS.model_type))
print('Assuming dev set located at data/dev.tag')
ARGS.dev_set_path = 'data/dev.tag'

# Parse data sets into word and tag sequences
TRAIN_X_RAW, TRAIN_Y = helpers.parse_data(ARGS.training_set_path)
DEV_X_RAW, DEV_Y = helpers.parse_data(ARGS.dev_set_path)
TEST_X_RAW, TEST_Y = helpers.parse_data(ARGS.test_set_path)
print(TRAIN_Y)
print('Proportion of GENE in training set', sum(TRAIN_Y)/len(TRAIN_Y))
print('Proportion of GENE in dev set', sum(DEV_Y)/len(DEV_Y))

# Convert data sets to feature representations
# TODO: word_vectorizer - stopwords?
# TODO: char_vectorizer - or char_wb?, ngram_range, 
N_WORDS = [0] #[0, 1, 2, 3]
N_CHARS = [0] #[0, 1, 2, 3]
N_TAGS = [0] #[0, 1, 2, 3]
for n_words, n_chars, n_tags in itertools.product(N_WORDS, N_CHARS, N_TAGS):
    print('Training with n_words=' + str(n_words) + ' n_chars=' + str(n_chars) + ' n_tags=' + str(n_tags))
    print('Generating feature vectors for training set...')
    word_vectorizer = CountVectorizer(analyzer='word', binary=True)
    word_vectorizer.fit(TRAIN_X_RAW)
    char_vectorizer = CountVectorizer(analyzer='char', binary=True, lowercase=False, ngram_range=(1, 4))
    char_vectorizer.fit(TRAIN_X_RAW)
    num_samples = len(TRAIN_Y)
    train_x = helpers.extractMaxEntFeatures(TRAIN_X_RAW, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples)