import os
import numpy as np


# HMM algorithm
class HMM_Cell(object):
    def __init__(self, config):
        self.n_classes = config.n_classes
        self.n_feature = config.n_feature

        # init define three probability
        self.init_probability = np.zeros((self.n_classes, 1))
        self.transition = np.zeros((self.n_classes, self.n_classes))
        self.emission = np.zeros((self.n_feature, self.n_classes))

    def fit(self, x, y):
        """
        using train data and train target to fit three probabilities
        :param x: example x is like [[12, 33, 44], [22, 332], [231]]
        :param y: example y is like [[1, 2, 3], [0, 2], [3]]
        :return: None
        """
        for i in range(len(x)):
            for j in range(len(x[i])):

                # init probability
                if j == 0:
                    self.init_probability[y[i][j]] += 1

                # transition probability
                else:
                    self.transition[y[i][j], y[i][j - 1]] += 1

                # emission probability
                self.emission[x[i][j]][y[i][j]] += 1

        # laplace smoothing
        self.init_probability = (self.init_probability + 1) / np.sum(self.init_probability)
        self.transition = (self.transition + 1) / (np.sum(self.transition) + self.n_classes)
        self.emission = (self.emission + 1) / (np.sum(self.emission) + self.n_feature)

    # viterbi algorithm(without start node and stop node)
    def predict_label(self, x):
        """
        using Viterbi algorithm and probabilities to calculate max score path
        :param x: example x is like [[12, 33, 44], [22, 332], [231]]
        :return: predicted label, the shape is same as x
        """
        best_path_ids, label_col = [], []
        for i in range(len(x)):
            # forward_var denote end this step each node will get max score
            forward_var = np.log(self.init_probability)

            for j in range(len(x[i])):
                if j == 0:
                    forward_var += np.log(self.emission[x[i][j]].reshape(-1, 1))

                else:
                    current_score, current_ids = np.zeros((self.n_classes, 1)), np.zeros(self.n_classes, dtype=int)
                    for next_tag in range(self.n_classes):
                        next_score = forward_var + np.log(self.transition[next_tag].reshape(-1, 1))
                        max_score_id = np.argmax(next_score.reshape(-1))

                        current_score[next_tag] = np.max(next_score)
                        current_ids[next_tag] = max_score_id

                    # update forward_var with each step
                    forward_var = current_score + np.log(self.emission[x[i][j]].reshape(-1, 1))
                    best_path_ids.append(current_ids.tolist())

            start_node = np.argmax(forward_var)
            label_col.append(self.find_path(start_node, best_path_ids))
            best_path_ids = []

        return label_col

    def find_path(self, start_node, best_path_ids):
        label = [start_node]
        for i in range(len(best_path_ids) - 1, -1, -1):
            label.append(best_path_ids[i][label[-1]])
        label.reverse()
        return label
