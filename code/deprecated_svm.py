# Adapted from https://rpubs.com/Sharon_1684/454441

import sys
import util
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC


def main(vis = False):
    X, Y = util.load_dataset('../FinalPhotosData', ['mask_weared_incorrect', 'with_mask', 'without_mask'], 224)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    svm = SVC(kernel='rbf', probability=True, random_state=42)

    # fit model
    svm.fit(X_train, Y_train)

    # predictedTrain = clf.predict(X_train)
    # util.findF1Score(predictedTrain, Y_train, "On Train Set")
    Y_pred = svm.predict(X_test)

    # calculate accuracy
    accuracy = accuracy_score(Y_test, Y_pred)
    print('Model accuracy is: ', accuracy)

    # # predictedTest = clf.predict(X_test)
    # # util.findF1Score(predictedTest, Y_test, "On Test Set")
    # accOnTest = clf.score(X_test, Y_test)
    # print("Acc on test: ", accOnTest)

    if vis: util.visualize(X_train, Y_train)


if __name__ == "__main__":
    main(True)