from __future__ import division
import re
import subprocess
from operator import add
import numpy as np
from scipy import sparse
import time
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import random

F1_REGEX = '(.*)(Precision: )(0.[0-9]*)( Recall: )(0.[0-9]*)( F1: )(0.[0-9]*)'

def parse_data(data_set_path):
    identifiers = []
    X = []
    Y = []
    with open(data_set_path, 'r') as f:
        for line_num, line in enumerate(f):
            line = line.strip()
            if (line_num % 2 == 1):
                line = re.split('_| ', line)
                # Process words
                X.append(' '.join(line[::2]))
                # Process tags
                Y.append(line[1::2])
            else:
                identifiers.append(line)

    Y_features = []
    for tag_sequence in Y:
        for tag in tag_sequence:
            if (tag == 'TAG'):
                Y_features.append(0)
            else:
                Y_features.append(1)
    Y_features = np.array(Y_features)
    return (identifiers, X, Y_features)

def extractMaxEntFeatures(X, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples, cache=None, predicted_ys=[]):
    # Process X with the following features
    # one hot vector of training set words
    # one hot vector of words in english?
    # previous 1, 2, 3 words
    # previous 1, 2, 3 tags
    if cache is not None and os.path.isfile('cache/' + cache + '.npz'):
        loader = np.load('cache/' + cache + '.npz')
        return sparse.csr_matrix((loader['data'], loader['indices'], \
                                  loader['indptr']), shape = loader['shape'])
    else:
        word_vocabulary_size = len(word_vectorizer.vocabulary_)
        char_vocabulary_size = len(char_vectorizer.vocabulary_)
        num_features = (n_words + 1) * word_vocabulary_size + \
                    1 * char_vocabulary_size + \
                    3 * n_tags
        X_features = []
        count = 0
        sequence_start_index = 0
        s = time.time()
        for word_sequence in X:
            word_sequence = word_sequence.split(' ')
            for word_index in range(len(word_sequence)):
                x = featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features, predicted_ys=predicted_ys[sequence_start_index:sequence_start_index+len(word_sequence)])
                X_features.append(x)
                count += 1
                if count % 20000 == 0:
                    s = time.time()
            sequence_start_index += len(word_sequence)
        s = time.time()
        X_features = sparse.vstack(X_features).tocsr()
        try:
            if cache is not None:
                if not os.path.exists('cache/'):
                    os.makedirs('cache/')
                np.savez('cache/' + cache + '.npz', data=X_features.data, indices=X_features.indices, \
                        indptr=X_features.indptr, shape=X_features.shape)
        except OSError as e:
            print('OSError occured when trying to create cache folder... not caching...')
        return X_features

# ends in -ase or -ases
# ends in -ine or -ines
# ends in -ate or -ates
# ends in -gen or -gens
# ends in -in or -ins
# ends in -ic or -ics
# ends in -yme or -ymes
# first letter uppercase
# all letters uppercase
# mix of upper and lowercase
# contains numbers and letters
# previous was - / next is -???

def featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features, predicted_ys=[]):
    x = []
    # process word level features
    for index in range(word_index-n_words, word_index+1):
        if index < 0:
            x.append(np.zeros((1, word_vocabulary_size)))
        else:
            word = word_sequence[index]
            vector = word_vectorizer.transform([word])
            x.append(vector)

    # process char level features
    min_word_index_for_n_chars = max(word_index-n_chars, 0)
    words = ' '.join(word_sequence[min_word_index_for_n_chars:word_index+1])
    chars_vector = char_vectorizer.transform([words])
    x.append(chars_vector)

    # process previous tags
    for index in range(word_index-n_tags, word_index):
        if index < 0:
            x.append(np.array([1, 0, 0]))
        else:
            if predicted_ys[index] == 0:
                x.append(np.array([0, 1, 0]))
            else:
                x.append(np.array([0, 0, 1]))
    return sparse.hstack(x)

def accuracy(predicted_y, actual_y):
    return np.sum(predicted_y == actual_y)/len(actual_y)

def evaluateLogisticRegressionModel(dev_x_raw, dev_identifiers, dev_predicted_y):
    front = './outputs/output' + str(random.randint(0,100000000))
    tags_filename = front + '.tag'
    with open(tags_filename, 'w') as f:
        index = 0
        for word_sequence, identifier in zip(dev_x_raw, dev_identifiers):
            f.write(identifier+'\n')
            word_sequence = word_sequence.split(' ')
            tagged_sequence = ""
            for word in word_sequence:
                if dev_predicted_y[index] == 0:
                    tagged_sequence += (word + '_' + 'TAG ')
                else:
                    tagged_sequence += (word + '_' + 'GENE1 ')
                index += 1
            f.write(tagged_sequence[:-1]+'\n')

    formatted_output = subprocess.check_output(['perl', 'eval/format.perl', tags_filename])
    output_filename = front + '.format'
    with open(output_filename, 'w') as f:
        f.write(formatted_output.decode('utf-8'))
    evaluation = subprocess.check_output(['perl', 'eval/alt_eval.perl', 'data/dev.gold', output_filename, 'data/Correct.data']).decode('utf-8')
    evaluation_results = re.findall(F1_REGEX, evaluation)[0]
    dev_p = evaluation_results[2]
    dev_r= evaluation_results[4]
    dev_f1_score = evaluation_results[6]
    return dev_p, dev_r, dev_f1_score

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,\
                            num_layers=1, batch_first=True, nonlinearity='relu')
        self.W_o = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()
    
    def forward(self, all_x, h0=None):
        h0_orig = h0
        if h0 is None:
            h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        output, h_n = self.rnn(all_x, h0)
        last_out = self.log_softmax(self.W_o(h_n.squeeze(0)))
        if h0_orig is None:
            all_out = [self.log_softmax(self.W_o(output[:,b,:])) for b in range(30)]
        else:
            all_out = None
        return all_out, last_out, h_n

CRITERION = torch.nn.NLLLoss(weight=torch.Tensor([1, 10]))
def train(model, xs, ys, optimizer):
    all_out, last_out, _ = model(xs)
    for index, out in enumerate(all_out):
        model.zero_grad()
        loss = CRITERION(out, ys[:,index].long())
        loss.backward(retain_graph=True)
        optimizer.step()
    return loss
