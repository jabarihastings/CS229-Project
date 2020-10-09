import sys
import util
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# sys.path.append('../data')

def main():
    X, Y = util.load_dataset('../data/small/')

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)
    clf = LogisticRegression(random_state=0, max_iter=1000).fit(X_train, Y_train)

    #Y_predict = clf.predict(X_test)
    acc = clf.score(X_test, Y_test)
    print(acc)

    util.visualize(X, Y)
    return 0



if __name__ == "__main__":
    main()