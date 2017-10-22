import re
import numpy as np

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
    return (X, Y)

def extractMaxEntFeatures(X, Y, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags):
    # Process Y
    Y_features = []
    for tag_sequence in Y:
        for tag in tag_sequence:
            if (tag == 'TAG'):
                Y_features.append(0)
            else:
                Y_features.append(1)
    Y_features = np.array(Y_features)

    # Process X with the following features
    # one hot vector of training set words
    # one hot vector of words in english?
    # previous 1, 2, 3 words
    # previous 1, 2, 3 tags
    num_samples = len(Y_features)
    word_vocabulary_size = len(word_vectorizer.vocabulary_)
    char_vocabulary_size = len(char_vectorizer.vocabulary_)
    num_features = (n_words + 1) * word_vocabulary_size + \
                   n_chars * char_vocabulary_size + \
                   n_tags
    print(num_samples, num_features)
    X_features = np.zeros((num_samples, num_features))
    count = 0
    for word_sequence in X:
        for word_index in range(len(word_sequence)):
            x = featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features)
            X_features[count] = x
            count += 1
    return (X_features, Y_features)

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
    x = np.zeros((1, num_features))
    for index in range(word_index-n_words, word_index+1):
        if index >= 0:
            word = word_sequence[index]
            vector = word_vectorizer.transform([word])
            x[index*word_vocabulary_size:(index+1)*word_vocabulary_size] = vector
    return x