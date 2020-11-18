import os
import util
import deep_net_utils
from sklearn.linear_model import LogisticRegression
import random
import argparse
from sklearn.svm import SVC
import pickle


parser = argparse.ArgumentParser()
# setting the default to be 'experiments/fallback' so we don't risk overwriting existing experiments
parser.add_argument('--model_dir', default='experiments/fallback',
                    help="Directory containing params.json")
parser.add_argument('--verbose', default=1,
                    help="Directory containing params.json")
parser.add_argument('--vis', default=0,
                    help="Directory containing params.json")
# You may evaluate on 'train', 'val', or 'test'.
# Default '' means that we are simply training, and metrics for train and val will be saved. Model will also be saved
parser.add_argument('--evaluate', default='',
                    help="Directory containing params.json")

def evaluate(args, params):
    split = args.evaluate
    data_dir = '../data/' + params.dataset
    X, Y = util.load_dataset_from_split_directory(data_dir + '/' + split + '/', params.img_dimensions, args.verbose)

    # Load model
    saved_model_path = os.path.join(
        args.model_dir, "finalized_model.sav")
    loaded_model = pickle.load(open(saved_model_path, 'rb'))

    # predit
    predicted = loaded_model.predict(X)

    # evaluate metrics
    confus_save_path = os.path.join(
        args.model_dir, "confus_f1_{}.json".format(split))
    util.compute_and_save_f1(predicted, Y, confus_save_path)


def fit_softmax_or_svm(X_train, Y_train, params):
    if params.model_type == 'softmax':
        return LogisticRegression(random_state=0, max_iter=params.iter, multi_class='multinomial', solver='lbfgs').fit(X_train, Y_train)
    if params.model_type == 'svm':
        svm = SVC(kernel=params.kernel, probability=True, random_state=42).fit(X_train, Y_train)
        return svm
    return None


def train(args, params):
    random.seed(229)


    # Check model_type is set (if 'model_type' is 'svm', 'kernel' is also set)
    assert(hasattr(params, 'model_type'))
    if params.model_type == 'svm': assert(hasattr(params, 'kernel'))

    data_container = util.load_dataset_for_split(args, params)
    X_train, Y_train = data_container['train']
    X_val, Y_val = data_container['val']
    X_test, Y_test = data_container['test']

    model = fit_softmax_or_svm(X_train, Y_train, params)

    model_file_path = args.model_dir + 'finalized_model.sav'
    pickle.dump(model, open(model_file_path, 'wb'))

    predicted_train = model.predict(X_train)
    predicted_val = model.predict(X_val)

    train_confus_save_path = os.path.join(
        args.model_dir, "confus_f1_train.json")
    val_confus_save_path = os.path.join(
        args.model_dir, "confus_f1_val.json")

    util.compute_and_save_f1(predicted_train, Y_train, train_confus_save_path)
    util.compute_and_save_f1(predicted_val, Y_val, val_confus_save_path)


    train_accur = model.score(X_train, Y_train)
    print("Acc on train: ", train_accur)

    val_accur = model.score(X_val, Y_val)
    print("Acc on validation: ", val_accur)

    if args.vis: util.visualize(X_train, Y_train)


if __name__ == "__main__":
    random.seed(229)
    args = parser.parse_args()

    # load hyperparameters
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = deep_net_utils.Params(json_path)

    if args.evaluate == "":
        train(args, params)
    else:
        evaluate(args, params)