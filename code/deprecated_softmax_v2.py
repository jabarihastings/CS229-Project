import os
import util
import deep_net_utils
from sklearn.linear_model import LogisticRegression
import random
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', default='experiments/softmax_kaggle_baseline',
                    help="Directory containing params.json")
parser.add_argument('--verbose', default=1,
                    help="Directory containing params.json")
parser.add_argument('--vis', default=0,
                    help="Directory containing params.json")

def load_dataset(args, params):
    # Get data_dir
    data_dir = '../data/' + params.dataset
    X_train, Y_train = util.load_dataset_from_split_directory(data_dir + '/train/', params.img_dimensions, args.verbose)
    X_val, Y_val = util.load_dataset_from_split_directory(data_dir + '/val/', params.img_dimensions, args.verbose)
    X_test, Y_test = util.load_dataset_from_split_directory(data_dir + '/test/', params.img_dimensions, args.verbose)

    return {
        'train': (X_train, Y_train),
        'val': (X_val, Y_val),
        'test': (X_test, Y_test)
    }


def main(args):
    random.seed(229)
    # load hyperparameters
    json_path = os.path.join(args.model_dir, "params.json")
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = deep_net_utils.Params(json_path)

    data_container = load_dataset(args, params)
    X_train, Y_train = data_container['train']
    X_val, Y_val = data_container['val']
    X_test, Y_test = data_container['test']

    clf = LogisticRegression(random_state=0, max_iter=params.iter, multi_class='multinomial', solver='lbfgs').fit(X_train, Y_train)

    predicted_train = clf.predict(X_train)
    predicted_val = clf.predict(X_val)

    train_confus_save_path = os.path.join(
        args.model_dir, "confus_f1_train.json")
    val_confus_save_path = os.path.join(
        args.model_dir, "confus_f1_val.json")

    util.compute_and_save_f1(predicted_train, Y_train, train_confus_save_path)
    util.compute_and_save_f1(predicted_val, Y_val, val_confus_save_path)


    train_accur = clf.score(X_train, Y_train)
    print("Acc on train: ", train_accur)

    val_accur = clf.score(X_val, Y_val)
    print("Acc on validation: ", val_accur)

    if args.vis: util.visualize(X_train, Y_train)


if __name__ == "__main__":
    main(parser.parse_args())