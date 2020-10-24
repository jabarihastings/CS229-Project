import numpy as np

from os import listdir
from sklearn.decomposition import PCA
from skimage.transform import resize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def load_dataset(train_path, classes, img_dimensions, add_intercept=False, verbose=True):
    """
    Args:
        train_path: The folder containing the images
        add_intecept: Whether an intercept should be added
        verbose: Whether to print status
    
    Returns:
        X (n_examples, dim), Y (n_examples,)
    """
    x = []
    y = []

    if verbose:
        print("Loading data from path: {} ".format(train_path))

    for i, sfold in enumerate(classes):
        if verbose:
            print("Loading {} examples...".format(sfold))

        path = train_path + "/" + sfold + "/"
        for n, f in enumerate(listdir(path), start=1):
            if "png" not in f and "jpg" not in f: continue
            img = plt.imread(path + f)[:, :, 0] # Just one dimension
            img = resize(img, (img_dimensions, img_dimensions), anti_aliasing=True)
            x.append(img.flatten())
            y.append(i)

        if verbose:
            print("Loaded {} files.".format(n))

    if verbose:
        print("Done!")    
        
    return np.array(x), np.array(y)

def visualize(x, y):
    """Plots the first three principal components of data
    
    Args:
        x: data of shape (n_examples, dim)
        y: abels of data (n_examples)
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    pca = PCA(n_components=3)
    x_r = pca.fit_transform(x)

    colors = ['red', 'blue', 'purple'] # find better colors and ensure more than labels

    for i in range(len(colors)):
        ax.scatter(x_r[y == i, 0], x_r[y == i, 1], x_r[y == i, 2], color=colors[i],alpha=0.3)
    
    ax.set_title('PCA of Data Set')
    plt.show()

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
