import ipdb
import numpy
from sklearn.feature_extraction.text import   TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import zero_one_score
from sklearn.feature_selection import SelectKBest, chi2
from time import time
from sklearn.externals import joblib

#This is the code to organize data
def train_model():

###   Steps to get and store the tfidf values
#    vectorizer = TfidfVectorizer(stop_words='english', min_n=1, max_n=2,
#                               smooth_idf=True, sublinear_tf=True, max_df=0.5)

#    train_data = vectorizer.fit_transform(generate_emails(training_filenames))
#    test_data = vectorizer.transform(generate_emails(test_filenames))
#    joblib.dump(train_data.tocsr(), 'train_data.joblib')
#    joblib.dump(test_data.tocsr(), 'test_data.joblib')
#    joblib.dump(self.train_target, 'train_target.joblib')
#    joblib.dump(self.test_target, 'test_target.joblib')
###

    train_data = joblib.load('train_data.joblib', mmap_mode='c')
    test_data = joblib.load('test_data.joblib', mmap_mode='c')
    train_target = joblib.load('train_target.joblib', mmap_mode='c')
    test_target = joblib.load('test_target.joblib', mmap_mode='c')

###   Steps to select best features
#    print "Selecting K-best features by chi squared test"
#    start_time = time()
#    ch2 = SelectKBest(chi2, k=100)
#    train_data = ch2.fit_transform(train_data, train_target)
#    test_data = ch2.transform(test_data)
#    print "[Train data] n_samples: %d, n_features: %d" % train_data.shape
#    print "[Test data] n_samples: %d, n_features: %d" % test_data.shape
#    print "Done in %0.3fs" % (time() - start_time)
###
    if train_data.shape[0] == 0:
        print "train_data is empty. No vectors to train on."
        return None

    clf = LinearSVC() #SGDClassifier(n_iter=10, loss='modified_huber')
    print "Training %s" % (clf),
    start_time=time()
    clf.fit(train_data, train_target)
    train_time = time() - start_time
    print "Done in %0.3fs" % train_time

    print "Testing..."
    test_start = time()
    predicted = clf.predict(test_data)
    accuracy = zero_one_score(test_target, predicted)
    error_rate = 1 - accuracy
    test_time = time() - test_start
    print "Done in %0.3fs" % test_time

    print "Accuracy: ", numpy.mean(predicted == self.test_target)
    print "Z1 Accuracy: ", accuracy


def main():
    train_model()

if __name__ == '__main__':
    main()