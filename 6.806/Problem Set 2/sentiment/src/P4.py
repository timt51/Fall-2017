from __future__ import division
from collections import namedtuple
import gzip
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data_utils
import time

##################################################################
# Helper functions
##################################################################
def get_raw_data(file_name):
    f = open(file_name)
    x_raw = []
    y = []
    for line in f:
        x_raw.append(line[2:])
        y.append(int(line[0]))
    y = np.array(y)
    return x_raw, y

def accuracy(predicted_y, actual_y):
    return np.sum(predicted_y == actual_y)/len(actual_y)

##################################################################
# Load embeddings
##################################################################
EMBEDDING_SIZE = 300
VOCABULARY_SIZE = 12986
EMBEDDINGS = np.zeros((EMBEDDING_SIZE, VOCABULARY_SIZE))
VOCABULARY_INDEX = {}
with gzip.open('../word_vectors.txt.gz', 'rb') as f:
    for index, line in enumerate(f):
        line = line.split()
        EMBEDDINGS[:, index] = np.array(map(float, line[1:]))
        VOCABULARY_INDEX[line[0]] = index
##################################################################
# Load and store the training, dev, and data
##################################################################
TRAIN_X_RAW, TRAIN_Y = get_raw_data("../data/stsa.binary.train")
DEV_X_RAW, DEV_Y = get_raw_data("../data/stsa.binary.dev")
TEST_X_RAW, TEST_Y = get_raw_data("../data/stsa.binary.test")
# Convert the raw training data to a bag of words representation
VECTORIZER = CountVectorizer(vocabulary=VOCABULARY_INDEX)
VECTORIZER.fit(TRAIN_X_RAW)

TRAIN_X = VECTORIZER.transform(TRAIN_X_RAW).todense()
for index, row in enumerate(TRAIN_X):
    if np.sum(row) < 1e-2:
        TRAIN_X[index, :] = np.ones((1, VOCABULARY_SIZE),dtype=np.float32)
TRAIN_X = TRAIN_X / TRAIN_X.sum(axis=1)
TRAIN_X = np.matmul(EMBEDDINGS, TRAIN_X.T)

DEV_X = VECTORIZER.transform(DEV_X_RAW).T
DEV_X = DEV_X / DEV_X.sum(axis=0)
DEV_X = np.matmul(EMBEDDINGS, DEV_X)

TEST_X = VECTORIZER.transform(TEST_X_RAW).T
TEST_X = TEST_X / TEST_X.sum(axis=0)
TEST_X = np.matmul(EMBEDDINGS, TEST_X)

TRAIN_DATA = data_utils.TensorDataset(torch.from_numpy(TRAIN_X.T.astype(np.float32)), torch.from_numpy(TRAIN_Y))
TRAIN_LOADER = data_utils.DataLoader(TRAIN_DATA, batch_size=173,
                                          shuffle=True, num_workers=2, drop_last=True)
DEV_DATA = data_utils.TensorDataset(torch.from_numpy(DEV_X.T.astype(np.float32)), torch.from_numpy(DEV_Y))
DEV_LOADER = data_utils.DataLoader(DEV_DATA, batch_size=1,
                                          shuffle=False, num_workers=2)
TEST_DATA = data_utils.TensorDataset(torch.from_numpy(TEST_X.T.astype(np.float32)), torch.from_numpy(TEST_Y))
TEST_LOADER = data_utils.DataLoader(TEST_DATA, batch_size=1,
                                          shuffle=False, num_workers=2)
##################################################################
# Define the network
##################################################################
class SentimentNet(nn.Module):
    def __init__(self, hidden_dim):
        super(SentimentNet, self).__init__()
        self.linear1 = nn.Linear(EMBEDDING_SIZE, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        x = F.tanh(self.linear1(x))
        x = F.log_softmax(self.linear2(x))
        return x

##################################################################
# Parameters for the best result only
# HIDDEN_LAYER_DIMS = [EMBEDDING_SIZE/4, EMBEDDING_SIZE/2, EMBEDDING_SIZE, EMBEDDING_SIZE*2]
# HIDDEN_LAYER_DIMS = [int(x) for x in HIDDEN_LAYER_DIMS]
# LEARNING_RATES = [1E-5, 1E-3, 1E-1, 1E1]
# WEIGHT_DECAYS = [1E-5, 1E-3, 1E1]
##################################################################
HIDDEN_LAYER_DIMS = [EMBEDDING_SIZE*2]
HIDDEN_LAYER_DIMS = [int(x) for x in HIDDEN_LAYER_DIMS]
LEARNING_RATES = [1E-3]
WEIGHT_DECAYS = [1E-5]
CRITERION = nn.NLLLoss().cuda()
BATCH_SIZE = 173
EPOCHS = 50

def train(hidden_layer_dim, learning_rate, weight_decay):
    sentiment_net = SentimentNet(hidden_layer_dim).cuda()
    optimizer = optim.Adam(sentiment_net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    for epoch in range(EPOCHS):
        sentiment_net.train()
        for data in TRAIN_LOADER:
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = sentiment_net(inputs)
            loss = CRITERION(outputs, labels)
            loss.backward()
            optimizer.step()

    sentiment_net.eval()
    correct = 0
    total = 0
    for data in DEV_LOADER:
        ins, labels = data
        ins, labels = ins.cuda(), labels.cuda()
        outputs = sentiment_net(Variable(ins))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    dev_acc = correct / total

    sentiment_net.eval()
    correct = 0
    total = 0
    for data in TEST_LOADER:
        ins, labels = data
        ins, labels = ins.cuda(), labels.cuda()
        outputs = sentiment_net(Variable(ins))
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    test_acc = correct / total
    return dev_acc, test_acc

torch.manual_seed(77)
np.random.seed(77)
Result = namedtuple('Result', 'hidden_dim learning_rate weight_decay dev test')
RESULTS = []
for hidden_layer_dim in HIDDEN_LAYER_DIMS:
    for learning_rate in LEARNING_RATES:
        for weight_decay in WEIGHT_DECAYS:
            start = time.time()
            print('Hidden Layer Dimension: '+str(hidden_layer_dim)+' Learning Rate: '+str(learning_rate)+' Weight Decay: '+str(weight_decay))
            dev_acc, test_acc = train(hidden_layer_dim, learning_rate, weight_decay)
            print('Dev Accuracy: ', dev_acc)
            RESULTS.append(Result(hidden_layer_dim, learning_rate, weight_decay, dev_acc, test_acc))
            print('Time: ', time.time() - start)
RESULTS = sorted(RESULTS, key=lambda x: x.dev, reverse=True)
print('Best Result:', RESULTS[0])

##################################################################
# Graphs
# BEST_HIDDEN_DIM = RESULTS[0].hidden_dim
# BEST_LR = RESULTS[0].learning_rate
# BEST_WEIGHT_DECAY = RESULTS[0].weight_decay

# # Holding hidden dim and lr constant
# X = []
# Y = []
# for result in RESULTS:
#     if (result.hidden_dim == BEST_HIDDEN_DIM and result.learning_rate == BEST_LR):
#         X.append(result.weight_decay)
#         Y.append(result.dev)
# X = np.log10(X)
# X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
# plt.plot(X, Y)
# plt.axis([-6, 2, 0.45, 1])
# plt.title("Weight Decay vs Accuracy")
# plt.xlabel('log Weight Decay')
# plt.ylabel('Accuracy')
# plt.show()

# # Holding hidden dim and weight decay constant
# X = []
# Y = []
# for result in RESULTS:
#     if (result.hidden_dim == BEST_HIDDEN_DIM and result.weight_decay == BEST_WEIGHT_DECAY):
#         X.append(result.learning_rate)
#         Y.append(result.dev)
# X = np.log10(X)
# X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
# plt.plot(X, Y)
# plt.axis([-6, 2, 0.45, 1])
# plt.title("Learning Rate vs Accuracy")
# plt.xlabel('log Learning Rate')
# plt.ylabel('Accuracy')
# plt.show()

# # Holding learning rate and weight decay constant
# X = []
# Y = []
# for result in RESULTS:
#     if (result.learning_rate == BEST_LR and result.weight_decay == BEST_WEIGHT_DECAY):
#         X.append(result.hidden_dim)
#         Y.append(result.dev)
# X, Y = (list(t) for t in zip(*sorted(zip(X, Y))))
# plt.plot(X, Y)
# plt.axis([0, 650, 0.45, 1])
# plt.title("Hidden Layer Dimension vs Accuracy")
# plt.xlabel('Hidden Layer Dimension')
# plt.ylabel('Accuracy')
# plt.show()
##################################################################
