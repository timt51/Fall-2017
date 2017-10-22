from __future__ import division
import subprocess
import argparse
import re
import helpers
import itertools
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
else:
    # parameters for hyperparameter tuning
    N_WORDS = [0] #[0, 1, 2, 3]
    N_CHARS = [0] #[0, 1, 2, 3]
    N_TAGS = [0] #[0, 1, 2, 3]
    NGRAMS_MAX = [4] # [1, 2, 3, 4, 5]
F1_REGEX = '(.*)(Precision: )(0.[0-9]*)( Recall: )(0.[0-9]*)( F1: )(0.[0-9]*)'
for n_words, n_chars, n_tags, ngram_max in itertools.product(N_WORDS, N_CHARS, N_TAGS, NGRAMS_MAX):
    # Generate feature vectors for training data
    print('Training with n_words=' + str(n_words) + ' n_chars=' + str(n_chars) + ' n_tags=' + str(n_tags))
    print('Generating feature vectors for training set...')
    word_vectorizer = CountVectorizer(analyzer='word', binary=True)
    word_vectorizer.fit(TRAIN_X_RAW)
    char_vectorizer = CountVectorizer(analyzer='char', binary=True, lowercase=False, ngram_range=(1, ngram_max))
    char_vectorizer.fit(TRAIN_X_RAW)
    num_samples = len(TRAIN_Y)
    cache = 'train'
    train_x = helpers.extractMaxEntFeatures(TRAIN_X_RAW, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples, cache)

    # Train the logistic model
    print('Training...')
    lr = LogisticRegression()
    lr.fit(train_x, TRAIN_Y)
    train_predicted_y = lr.predict(train_x)
    train_accurary = helpers.accuracy(train_predicted_y, TRAIN_Y)
    print('Train accuracy: ' + str(train_accurary))
    word_feature_names = word_vectorizer.get_feature_names()
    coefs = np.abs(lr.coef_[0]).tolist()
    top10 = sorted(zip(word_feature_names, coefs), key=lambda x: x[1], reverse=True)[:10]
    print("Highest Impact Features (for model with stop words removed)")
    for top in top10:
        print(top)

    # Evaluate on the dev set
    print('Evaluating on dev set...')
    if ARGS.model_type == 1 or ARGS.model_type == 4:
        print('Generating features for dev set...')
        cache = 'dev'
        dev_x = helpers.extractMaxEntFeatures(DEV_X_RAW, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, num_samples, cache)
        dev_predicted_y = lr.predict(dev_x)
        dev_accuracy = helpers.accuracy(dev_predicted_y, DEV_Y)
        print('Dev accuracy: ' + str(dev_accuracy))

        print('Writing output to file...')
        with open('output.tag', 'w') as f:
            index = 0
            for word_sequence, identifier in zip(DEV_X_RAW, DEV_IDENTIFIERS):
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
            f.write(str(formatted_output,'utf-8'))
        evaluation = str(subprocess.check_output(['perl', 'eval/alt_eval.perl', 'data/dev.gold', 'output.format', 'data/Correct.data']), 'utf-8')
        dev_f1_score = re.findall(F1_REGEX, evaluation)[0][6]
        print(dev_f1_score)

    elif ARGS.model_type == 2 or ARGS.model_type == 5:
        print('Generating features for dev set and evaluating...')
        for word_sequence in dev_x:
            word_sequence = word_sequence.split(' ')
            predicted_ys = []
            for word_index in range(len(word_sequence)):
                x = helpers.featurize(word_index, word_sequence, word_vectorizer, char_vectorizer, n_words, n_chars, n_tags, word_vocabulary_size, char_vocabulary_size, num_features)
                predicted_y = lr.predict(x)
                predicted_ys.append(predicted_y)
    # Predict on the test set