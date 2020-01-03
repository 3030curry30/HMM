import os
import numpy as np


class Result_Eval(object):
    def __init__(self, config):
        self.n_classes = config.n_classes
        self.class_dict = config.invert_class_dict
        self.y = np.zeros(1)
        self.y_hat = np.zeros(1)

    def parse_label(self, label):
        index_dict, index, s_index = {}, 0, -1
        while index < len(label):
            if self.class_dict[label[index]] == "S":
                index_dict[index] = index
            elif self.class_dict[label[index]] == "B":
                s_index = index
            elif self.class_dict[label[index]] == "E":
                index_dict[s_index] = index
            index += 1
        return index_dict

    def eval_model(self, y, y_hat):
        truth_word, predict_word, correct_word = 0, 0, 0
        for i in range(len(y)):
            y_dict = self.parse_label(y[i])
            y_hat_dict = self.parse_label(y_hat[i])

            truth_word += len(y_dict)
            predict_word += len(y_hat_dict)

            for key, value in y_hat_dict.items():
                if key in y_dict and y_dict[key] == value:
                    correct_word += 1

        # calculate Precision, Recall, F_Measure
        Precision = correct_word / predict_word * 100
        Recall = correct_word / truth_word * 100
        F_Measure = (2 * Precision * Recall) / (Precision + Recall)

        print("Precision: {:.2f}".format(Precision))
        print("Recall: {:.2f}".format(Recall))
        print("F_Measure: {:.2f}".format(F_Measure))

