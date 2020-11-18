import numpy as np

from os import listdir
from sklearn.decomposition import PCA
from skimage.transform import resize
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics as sklearn_metrics

def load_dataset_from_split_directory(path, img_dimensions, verbose=True):
    x = []
    y = []

    for n, f in enumerate(listdir(path), start=1):
        if "png" not in f and "jpg" not in f: continue
        img = plt.imread(path + f)[:, :, 0]  # Just one dimension
        img = resize(img, (img_dimensions, img_dimensions), anti_aliasing=True)
        x.append(img.flatten())
        label = int(f[0])
        y.append(label)

    if verbose:
        print("Loaded {} files.".format(len(y)))

    if verbose:
        print("Loading data from path: {} ".format(path))

    return np.array(x), np.array(y)


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

def load_dataset_for_split(args, params):
    # Get data_dir
    data_dir = '../data/' + params.dataset
    X_train, Y_train = load_dataset_from_split_directory(data_dir + '/train/', params.img_dimensions, args.verbose)
    X_val, Y_val = load_dataset_from_split_directory(data_dir + '/val/', params.img_dimensions, args.verbose)
    X_test, Y_test = load_dataset_from_split_directory(data_dir + '/test/', params.img_dimensions, args.verbose)

    return {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test)
    }

def compute_and_save_f1(saved_outputs, saved_labels, file):
    conf_matrix, report = f1_metrics(saved_outputs, saved_labels)

    text_file = open(file, "wt")
    text_file.write('Confusion matrix: \n {}\n\n Classification Report: \n {}'.format(conf_matrix, report))

def f1_metrics(outputs, labels):
    confusion_matrix = sklearn_metrics.confusion_matrix(labels, outputs)
    classification_report = sklearn_metrics.classification_report(labels, outputs, digits=3)
    return confusion_matrix, classification_report
