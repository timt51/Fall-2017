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
                    print('Time to generate first ' + str(count) + ' sample features: ' + str(time.time() - s) + 's')
                    s = time.time()
            sequence_start_index += len(word_sequence)
        s = time.time()
        X_features = sparse.vstack(X_features).tocsr()
        print('Time to stack... ' + str(time.time()-s) + 's')
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
    print('Writing output to file...')
    with open('output.tag', 'w') as f:
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

    print('Evaluating with provided perl scripts...')
    formatted_output = subprocess.check_output(['perl', 'eval/format.perl', 'output.tag'])
    with open('output.format', 'w') as f:
        f.write(formatted_output.decode('utf-8'))
    evaluation = subprocess.check_output(['perl', 'eval/alt_eval.perl', 'data/dev.gold', 'output.format', 'data/Correct.data']).decode('utf-8')
    evaluation_results = re.findall(F1_REGEX, evaluation)[0]
    dev_f1_score = evaluation_results[6]
    print('Dev Results: ' + reduce(add, evaluation_results))
    print('Dev F1 Score: ' + str(dev_f1_score))
    return dev_f1_score

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size
        self.batch_size = batch_size

        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size,\
                            num_layers=1, batch_first=True)
        self.W_o = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax()
    
    def forward(self, all_x):
        h0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size))
        output, h_n = self.rnn(all_x, h0)
        h_n = h_n.squeeze(0)
        last_out = self.log_softmax(self.W_o(h_n))
        all_out = [self.log_softmax(self.W_o(x)) for x in output]
        return all_out, last_out

def train(model, xs, ys, optimizer):
    model.zero_grad()
    _, last_out = model(xs)
    loss = F.nll_loss(last_out, ys[:,-1].long())
    loss.backward()
    optimizer.step()
    return loss
