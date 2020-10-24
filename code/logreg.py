import sys
import util
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier

def decisionMatrix(predicted, actual, verbose = None):
    isEqual = np.where(predicted != actual, 0, 1)
    tp = predicted.T.dot(isEqual)
    fp = (1-isEqual).T.dot(predicted)
    fn = (1-predicted).T.dot(1-isEqual)
    tn = (1-predicted).T.dot(isEqual)
    if (verbose):
        print(verbose)
        print("FP: %d" % fp)
        print("TP: %d" % tp)
        print("TN: %d" % tn)
        print("FN: %d" % fn)
    return (tp, fp, fn, tn)

def findF1Score(predicted, actual, verbose = None):
    (tp, fp, fn, tn) = decisionMatrix(predicted, actual, verbose)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)
    if verbose:
        print("Precision: ", precision)
        print("Recall: ", recall)
        print("F1: ", f1)
    return f1

def main():
    X, Y = util.load_dataset('../data/small/')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, Y_train)
    # clf = SGDClassifier(loss='log', max_iter=10000).fit(X_train, Y_train)

    predictedTrain = clf.predict(X_train)
    findF1Score(predictedTrain, Y_train, "On Train Set")
    accOnTrain = clf.score(X_train, Y_train)
    print("Acc on train: ", accOnTrain)

    predictedTest = clf.predict(X_test)
    findF1Score(predictedTest, Y_test, "On Test Set")
    accOnTest = clf.score(X_test, Y_test)
    print("Acc on test: ", accOnTest)

    util.visualize(X_train, Y_train)



if __name__ == "__main__":
    main()