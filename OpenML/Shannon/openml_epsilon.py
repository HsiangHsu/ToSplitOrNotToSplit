# -*- coding: utf-8 -*-
""""
Codes for ICML'20 on OpenML
Author: Hsiang Hsu
email:  hsianghsu@g.harvard.edu
"""
import tensorflow as tf
import pickle
import gzip
import numpy as np
from time import localtime, strftime
from sklearn.linear_model import LogisticRegression

# Load Data
pickle_file = 'datasets_parsed.pickle'
with open(pickle_file, "rb") as input_file:
    datasets = pickle.load(input_file)

filename = 'openml_epsilon_log_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())+'.txt'
file = open(filename,'w')

for i in range(47):
    if i in [26]:
        continue;
    file.write('=== Dataset {} ===\n'.format(i))
    file.flush()

    X0, X1 = datasets[i]["X0"], datasets[i]["X1"]
    Y0, Y1 = datasets[i]["Y0"], datasets[i]["Y1"]
    # train/test split
    def train_test_split(x, y, p):
        assert (len(x) == len(y))
        msk = np.random.rand(len(x)) < p
        train_x, train_y = x[msk], y[msk]
        test_x, test_y = x[~msk], y[~msk]
        return train_x, train_y, test_x, test_y

    p = .7
    train_X0, train_Y0, test_X0, test_Y0 = train_test_split(X0, Y0, p)
    train_X1, train_Y1, test_X1, test_Y1 = train_test_split(X1, Y1, p)

    # Training
    LR0 = LogisticRegression(solver = 'liblinear').fit(train_X0, train_Y0)
    LR1 = LogisticRegression(solver = 'liblinear').fit(train_X1, train_Y1)

    train_X = np.concatenate((train_X0, train_X1), axis=0)
    train_Y = np.concatenate((train_Y0, train_Y1), axis=0)
    test_X = np.concatenate((test_X0, test_X1), axis=0)
    test_Y = np.concatenate((test_Y0, test_Y1), axis=0)

    LR = LogisticRegression(solver = 'liblinear').fit(train_X, train_Y)

    # Accuracies
    acc_0_train = LR0.score(train_X0, train_Y0)
    acc_1_train = LR1.score(train_X1, train_Y1)

    acc_0_test = LR0.score(test_X0, test_Y0)
    acc_1_test = LR1.score(test_X1, test_Y1)

    acc_train = LR.score(train_X, train_Y)
    acc_test = LR.score(test_X, test_Y)
    file.write('Training Accuracies\n Decoupling0: {:.4f}\n Decoupling1:{:.4f}\n Group Blind{:.4f}\n'.format(acc_0_train, acc_1_train, acc_train))
    file.write('Test Accuracies\n Decoupling0: {:.4f}\n Decoupling1:{:.4f}\n Group Blind{:.4f}\n'.format(acc_0_test, acc_1_test, acc_test))
    file.flush()

    # disagreement between two classifiers
    disagreement0 = np.abs(LR0.predict(test_X0) - LR1.predict(test_X0)).mean()
    disagreement1 = np.abs(LR0.predict(test_X1) - LR1.predict(test_X1)).mean()
    file.write('Disagreement of 0 Classifier" {:.4f}\n'.format(disagreement0))
    file.write('Disagreement of 1 Classifier" {:.4f}\n'.format(disagreement1))
    file.flush()

    # epsilon_coupling
    decoupling0 = np.abs(LR0.predict(X0) - Y0).mean()
    decoupling1 = np.abs(LR1.predict(X1) - Y1).mean()
    group_blind0 = np.abs(LR.predict(test_X0) - test_Y0).mean()
    group_blind1 = np.abs(LR.predict(test_X1) - test_Y1).mean()
    eps_split = max(max(group_blind0, group_blind1) - max(decoupling0, decoupling1), 0)
    file.write('epsilon_coupling: {:.4f}\n'.format(eps_split))
    file.flush()

file.close()
