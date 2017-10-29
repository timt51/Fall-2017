from __future__ import division
import subprocess
import time
import argparse
import re
import helpers
import itertools
from functools import reduce
from operator import add
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import torch
import torch.utils.data as data_utils
from torch.autograd import Variable
import torch.optim as optim
from scipy import sparse
from tqdm import tqdm

# Create parser for input arguments
PARSER = argparse.ArgumentParser(description='Gene tagging with RNN model')
PARSER.add_argument('training_set_path', help='path of training set')
PARSER.add_argument('test_set_path', help='path of test set')

# Parse arguments and display the parsed arguments
print('Beginning gene tag training and evaluation...')
ARGS = PARSER.parse_args()
print('Using training set at ' + ARGS.training_set_path)
print('Using test set at ' + ARGS.test_set_path)
print('Assuming dev set located at data/dev.tag')
ARGS.dev_set_path = 'data/dev.tag'

# Parse data sets into word and tag sequences
TRAIN_IDENTIFIERS, TRAIN_X_RAW, TRAIN_Y = helpers.parse_data(ARGS.training_set_path)
DEV_IDENTIFIERS, DEV_X_RAW, DEV_Y = helpers.parse_data(ARGS.dev_set_path)
TEST_IDENTIFIERS, TEST_X_RAW, TEST_Y = helpers.parse_data(ARGS.test_set_path)
print('Proportion of GENE in training set', sum(TRAIN_Y)/len(TRAIN_Y))
print('Proportion of GENE in dev set', sum(DEV_Y)/len(DEV_Y))

# Run the model
N_WORDS = [0]
N_CHARS = [0]
N_TAGS = [0]
NGRAMS_MAX = [4]
HIDDEN_DIM = 100
BATCH_SIZE = 32
MAX_EPOCHS = 10
MAX_SEQ_LENGTH = 30
LEARNING_RATE = 1E-3
WEIGHT_DECAY = 1E-5
TO_CACHE = True
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
    train_x_merged = helpers.extractMaxEntFeatures(TRAIN_X_RAW, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples, cache=cache, predicted_ys=TRAIN_Y)
    train_x = []
    train_y = []
    word_vocabulary_size = len(word_vectorizer.vocabulary_)
    char_vocabulary_size = len(char_vectorizer.vocabulary_)
    num_features = (n_words + 1) * word_vocabulary_size + \
                    1 * char_vocabulary_size + \
                    3 * n_tags
    index = 0
    for word_sequence in TRAIN_X_RAW:
        length = len(word_sequence.split(' '))
        seq_matrix = train_x_merged[index:index+length]
        tags = TRAIN_Y[index:index+length]
        if length <= MAX_SEQ_LENGTH:
            empty = sparse.csr_matrix((MAX_SEQ_LENGTH-length, num_features))
            cut_matrix = sparse.vstack([seq_matrix, empty])
            cut_tags = np.zeros(MAX_SEQ_LENGTH)
            cut_tags[:length] = tags
        else:
            cut_matrix = seq_matrix[:MAX_SEQ_LENGTH, :]
            cut_tags = tags[:MAX_SEQ_LENGTH]
        train_x.append(cut_matrix)
        train_y.append(cut_tags)
        index += length
    train_x = np.array(train_x)
    train_y = np.array(train_y)

    # Create the RNN model
    output_size = 2
    model = helpers.RNN(num_features, HIDDEN_DIM, output_size, BATCH_SIZE).cuda()

    # Train the RNN model
    print('Training...')
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for epoch in range(MAX_EPOCHS):
        # shuffle trainx and trainy
        p = np.random.permutation(len(train_y))
        train_x, train_y = train_x[p], train_y[p]
        for batch_start_index in tqdm(range(0,len(train_y),BATCH_SIZE)):
            batch_train_x = train_x[batch_start_index:batch_start_index+BATCH_SIZE]
            batch_train_y = train_y[batch_start_index:batch_start_index+BATCH_SIZE]
            # s = time.time()
            xs = Variable(torch.from_numpy(np.array([x.todense() for x in batch_train_x]).astype(np.float32))).cuda()
            ys = Variable(torch.from_numpy(batch_train_y.astype(np.float32))).cuda()
            # print("time to convert...", s - time.time())
            loss = helpers.train(model, xs, ys, optimizer)
        print(epoch, loss)

    # Evaluate on the dev set
    print('Generating dev set predictions...')
    predicted_ys = []
    for word_sequence in DEV_X_RAW:
        word_sequence = word_sequence.split(' ')
        predicted_sequence_ys = []
        for word_index in range(len(word_sequence)):
            x = helpers.featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features, predicted_ys=predicted_sequence_ys)
            x = Variable(torch.from_numpy(np.expand_dims(x.todense(),axis=0).astype(np.float32)))
            output, hidden = model(x, hidden)
            _, predicted_y = torch.max(output, 0)
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
    print('Dev F1 score: ' + str(dev_f1_score))
    # RESULTS.append(Result(n_words, n_chars, n_tags, ngram_max, dev_f1_score))
