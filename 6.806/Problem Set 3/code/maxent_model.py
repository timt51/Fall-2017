import argparse
import helpers
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
TRAIN_X_RAW, TRAIN_Y_RAW = helpers.parse_data(ARGS.training_set_path)
DEV_X_RAW, DEV_Y_RAW = helpers.parse_data(ARGS.dev_set_path)
TEST_X_RAW, TEST_Y_RAW = helpers.parse_data(ARGS.test_set_path)

# Convert data sets to feature representations
# X = (n_samples, n_features), Y = (n_samples,) [TAG, GENE]
# Count TAG/GENE ratio in each data set and display ratios
WORD_VECTORIZER = CountVectorizer(analyzer='word')
WORD_VECTORIZER.fit(TRAIN_X_RAW) # TODO:
CHAR_VECTORIZER = CountVectorizer(analyzer='char') # TODO: or char_wb?
CHAR_VECTORIZER.fit(TRAIN_X_RAW) # TODO: 
helpers.extractMaxEntFeatures(TRAIN_X_RAW, TRAIN_Y_RAW, WORD_VECTORIZER, CHAR_VECTORIZER, 0, 0, 0)