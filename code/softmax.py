import sys
import util
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier


def main(vis = False):
    X, Y = util.load_dataset('../kaggleData/sorted', ['mask_weared_incorrect', 'with_mask', 'without_mask'], 30)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = LogisticRegression(random_state=0, max_iter=1000, multi_class='multinomial', solver='lbfgs').fit(X_train, Y_train)

    # predictedTrain = clf.predict(X_train)
    # util.findF1Score(predictedTrain, Y_train, "On Train Set")
    accOnTrain = clf.score(X_train, Y_train)
    print("Acc on train: ", accOnTrain)

    # predictedTest = clf.predict(X_test)
    # util.findF1Score(predictedTest, Y_test, "On Test Set")
    accOnTest = clf.score(X_test, Y_test)
    print("Acc on test: ", accOnTest)

    if vis: util.visualize(X_train, Y_train)


if __name__ == "__main__":
    main(True)