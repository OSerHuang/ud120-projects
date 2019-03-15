#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

from sklearn.svm import SVC

print "---------linear kernel with full size----------"
clf = SVC(kernel='linear')
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
accuracy = clf.score(features_test, labels_test)
t2 = time()
print "Training time:", t1 - t0, 's'
print "Predicting time:", t2 - t1, 's'
print "Accuracy:", accuracy

features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
print "--------linear kernel with reduced size---------"
clf = SVC(kernel='linear')
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
accuracy = clf.score(features_test, labels_test)
t2 = time()
print "Training time:", t1 - t0, 's'
print "Predicting time:", t2 - t1, 's'
print "Accuracy:", accuracy

print "---------------rbf kernel---------------"
clf = SVC(kernel='rbf')
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
accuracy = clf.score(features_test, labels_test)
t2 = time()
print "Training time:", t1 - t0, 's'
print "Predicting time:", t2 - t1, 's'
print "Accuracy:", accuracy

for c in [10**i for i in range(1, 5)]:
    print "------------------------------------"
    print "C:", c
    clf = SVC(kernel='rbf', C=c)
    t0 = time()
    clf.fit(features_train, labels_train)
    t1 = time()
    accuracy = clf.score(features_test, labels_test)
    t2 = time()
    print "Training time:", t1 - t0, 's'
    print "Predicting time:", t2 - t1, 's'
    print "Accuracy:", accuracy
    print "------------------------------------"

print "----------full size with optimized C-----------"
features_train, features_test, labels_train, labels_test = preprocess()
clf = SVC(kernel='rbf', C=10000)
t0 = time()
clf.fit(features_train, labels_train)
t1 = time()
accuracy = clf.score(features_test, labels_test)
t2 = time()
print "Training time:", t1 - t0, 's'
print "Predicting time:", t2 - t1, 's'
print "Accuracy:", accuracy

print "---------rbf kernel with reduced size----------"
features_train = features_train[:len(features_train)/100] 
labels_train = labels_train[:len(labels_train)/100] 
clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
print "10th:", predictions[10]
print "26th:", predictions[26]
print "50th:", predictions[50]

print "---------rbf kernel with full size----------"
features_train, features_test, labels_train, labels_test = preprocess()
clf = SVC(kernel='rbf', C=10000)
clf.fit(features_train, labels_train)
predictions = clf.predict(features_test)
print 'Total Chris(1) predictions:', sum(predictions)

#########################################################


