import os
import numpy
import argparse

import DataProcess
import Evaluation
import Config
import Model


def Terminal_parser():
    # define default train and test data
    source_path = "./data/people-daily.txt"
    train_path = "./data/train_data.txt"
    test_path = "./data/test_data.txt"

    parser = argparse.ArgumentParser()
    parser.description = "choose some parameters with terminal"

    parser.add_argument("--source", help='the path of source data', default=source_path)
    parser.add_argument("--train", help='the path of train data', default=train_path)
    parser.add_argument("--test", help='the path of test data', default=test_path)
    parser.add_argument("--n_class", help='the number of class', default=4)

    args = parser.parse_args()
    return args


def main():
    # using Terminal to get parameters
    config = Config.Config_Table(Terminal_parser())

    # get standard train and test file
    if not os.path.exists(config.train) or not os.path.exists(config.test):
        DataProcess.get_standard_file(config)

    # get char vocab
    vocab, index = {}, 0
    vocab, index = DataProcess.get_vocab(config.train, vocab, index)
    vocab, _ = DataProcess.get_vocab(config.test, vocab, index)
    config.n_feature = len(vocab)

    # get standard train and test data
    train_data, train_target = DataProcess.get_data(config.train, vocab, config.class_dict)
    test_data, test_target = DataProcess.get_data(config.test, vocab, config.class_dict)

    # define HMM model
    model = Model.HMM_Cell(config)
    model.fit(train_data, train_target)
    label = model.predict_label(test_data)

    result_eval = Evaluation.Result_Eval(config)
    result_eval.eval_model(test_target, label)


if __name__ == "__main__":
    main()