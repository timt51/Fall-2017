'''
Code to run EM for a latent topic model on text documents. Please note that
the code provided here is a only a template. You will need to fill in certain parts 
(labeled with TODOs) in order to complete it. 

You can run the code as:
python q_topic_em.py <data_directory> <num_topics> <num_iterations>
The sample data can be found in data/nyt/
'''

import sys, re, collections
from os import listdir
from os.path import isfile, join
import numpy as np
from operator import itemgetter
from sklearn.preprocessing import normalize

### Global structures to keep track of words and their mappings'''
word2Index = {}
vocabulary = []
vocabSize = 0
NUM_TOPICS = int(sys.argv[2])
NUM_DOCS = 0 

### Stopwords taken from NLTK
stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the', 'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll', 'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma', 'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

### Data is expected to be tokenized into words (separated by whitespace)'''
def readFile(filename):
	global vocabSize, vocabulary
	words = file(filename).read().lower().split()
	words = [w for w in words if w.isalpha() and w not in stopWords] #remove non-alpha and stopwords

	tokens = []

	### Create a mapping from words to indices. The EM algorithm will use these indices
	### in its data structures. 
	for w in words:
		if w not in word2Index:			
			word2Index[w] = vocabSize
			vocabulary.append(w)
			vocabSize += 1
		tokens.append(word2Index[w])

	return tokens 

### Reads an entire directory of files
def readDirectory(dirname):
	global NUM_DOCS
	fileData = [] # list of file data

	fileList = [ f for f in listdir(dirname) if isfile(join(dirname,f)) ]

	for f in fileList:
		fileData.append(readFile(join(dirname,f)))
		NUM_DOCS += 1

	return fileData

# --------------------------------------------------
def e_step(fileData, theta_t_z, theta_z_w):
	theta_w_z = np.transpose(theta_z_w)
	count_t_z = np.zeros([NUM_DOCS, NUM_TOPICS])
	count_w_z = np.zeros([vocabSize, NUM_TOPICS])

	### t is iterator over the documents {1, 2,..., n}
	for t in range(NUM_DOCS):

		''' In order to improve efficiency, we will go through each document 
				only once and calculate the posterior distributions as and when 
				necessary. So, the variable 'posterior_w_z[w][z]' below is implicitly
				representing P(z | w, t) for the current document t.

				The use of collections.defaultdict here is to calculate the 
				posteriors lazily, i.e. only for words that appear in 
				the current document.
		'''
		posterior_w_z = collections.defaultdict(lambda:np.zeros(NUM_TOPICS))

		### w is a word (in the form of a number corresponding to its index)
		### in the document
		for w in fileData[t]:
			
			if w not in posterior_w_z:
				'''TODO: calculate the posterior probability posterior_w_z[w][z]
					Make sure that the posterior_w_z[w] is a valid probability 
					distribution over the topics z.
				'''
				posterior_w_z[w] = (theta_t_z[t] * theta_w_z[w]) / np.dot(theta_t_z[t], theta_w_z[w])

			for z in range(NUM_TOPICS):
				'''TODO: Update soft counts count_t_z[t][z] '''
				count_t_z[t][z] += posterior_w_z[w][z]
				'''TODO: Update soft counts count_w_z[w][z] '''
				count_w_z[w][z] += posterior_w_z[w][z]
	return count_t_z, count_w_z


# --------------------------------------------------
def m_step(count_t_z, count_w_z):
	'''TODO: Get the max Likelihood estimate of theta_t_z[t][z]
		 Make sure that the theta_t_z[t] is a valid probability 
					distribution over the topics z.'''
	theta_t_z = normalize(count_t_z, axis=1, norm='l1')

	'''TODO: Get the max Likelihood estimate of theta_z_w[t][z]
	   Make sure that the theta_z_w[z] is a valid probability 
					distribution over the words.'''
	theta_z_w = normalize(count_w_z.transpose(), axis=1, norm='l1')

	return theta_t_z, theta_z_w


# --------------------------------------------------
def EM(fileData, num_iter):

	#Initialize parameters with random numbers
	theta_t_z = np.random.rand(NUM_DOCS, NUM_TOPICS)
	theta_z_w = np.random.rand(NUM_TOPICS, vocabSize)

	#normalize
	for t in range(NUM_DOCS):
		theta_t_z[t] /= np.sum(theta_t_z[t])
	for z in range(NUM_TOPICS):
		theta_z_w[z] /= np.sum(theta_z_w[z])


	for i in range(num_iter):
		print "Iteration", i+1, '...'
		count_t_z, count_w_z = e_step(fileData, theta_t_z, theta_z_w)
		theta_t_z, theta_z_w = m_step(count_t_z, count_w_z)

	return theta_t_z, theta_z_w


# --------------------------------------------------
if __name__ == '__main__':
	input_directory = sys.argv[1]
	fileData = readDirectory(input_directory)
	num_iter = int(sys.argv[3])

	print "Vocabulary:", vocabSize, "words."
	print "Running EM with", NUM_TOPICS, "topics."
	theta_t_z, theta_z_w = EM(fileData, num_iter)

	#Print out topic samples
	for z in range(NUM_TOPICS):
		
		wordProb = [(vocabulary[w], theta_z_w[z][w]) for w in range(vocabSize)]
		wordProb = sorted(wordProb, key = itemgetter(1), reverse=True)

		print "Topic", z+1
		for j in range(20):
			print wordProb[j][0], '(%.4f),' % wordProb[j][1], 
		print '\n'












