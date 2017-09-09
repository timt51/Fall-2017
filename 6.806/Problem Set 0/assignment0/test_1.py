from sklearn.datasets import fetch_20newsgroups
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

categories = ['alt.atheism', 'talk.religion.misc', 'comp.graphics', 'sci.space']
newsgroups_train = fetch_20newsgroups(subset='train',categories=categories)

# Modification for Assignment 1 Problem 1.2
# Performance before modification: 0.882135924027
# Performance after modification : 0.874995223641
newsgroups_train_data, _, newsgroups_train_targets, _ = train_test_split(newsgroups_train.data, newsgroups_train.target, test_size=0.5, random_state=42)


newsgroups_test = fetch_20newsgroups(subset='test',categories=categories)
vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(newsgroups_train_data)
vectors_test = vectorizer.transform(newsgroups_test.data)
clf = MultinomialNB(alpha=.01)
clf.fit(vectors, newsgroups_train_targets)
pred = clf.predict(vectors_test)
print "Performance:",metrics.f1_score(newsgroups_test.target, pred, average='macro')
