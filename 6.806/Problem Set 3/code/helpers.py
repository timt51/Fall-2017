import re
import numpy as np
from scipy import sparse
import time

def parse_data(data_set_path):
    X = []
    Y = []
    with open(data_set_path, 'r') as f:
        for line_num, line in enumerate(f):
            if (line_num % 2 == 1):
                line = line.strip()
                line = re.split('_| ', line)
                # Process words
                X.append(' '.join(line[::2]))
                # Process tags
                Y.append(line[1::2])

    Y_features = []
    for tag_sequence in Y:
        for tag in tag_sequence:
            if (tag == 'TAG'):
                Y_features.append(0)
            else:
                Y_features.append(1)
    Y_features = np.array(Y_features)
    return (X, Y_features)

def extractMaxEntFeatures(X, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples):
    # Process X with the following features
    # one hot vector of training set words
    # one hot vector of words in english?
    # previous 1, 2, 3 words
    # previous 1, 2, 3 tags
    word_vocabulary_size = len(word_vectorizer.vocabulary_)
    char_vocabulary_size = len(char_vectorizer.vocabulary_)
    num_features = (n_words + 1) * word_vocabulary_size + \
                   n_chars * char_vocabulary_size + \
                   n_tags
    X_features = sparse.lil_matrix((num_samples, num_features))
    count = 0
    s = time.time()
    for word_sequence in X:
        word_sequence = word_sequence.split(' ')
        for word_index in range(len(word_sequence)):
            x = featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features)
            X_features[count] = x
            count += 1
            if count % 20000 == 0:
                print('Time to generate first ' + str(count) + ' sample features: ' + str(time.time() - s) + 's')
                s = time.time()
    return X_features.tocsr()

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

def featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features):
    x = sparse.csr_matrix([])
    for index in range(word_index-n_words, word_index+1):
        if index < 0:
            x = sparse.hstack(np.zeros((1, word_vocabulary_size)))
        else:
            word = word_sequence[index]
            vector = word_vectorizer.transform([word])
            x = sparse.hstack((x, vector))
    return x