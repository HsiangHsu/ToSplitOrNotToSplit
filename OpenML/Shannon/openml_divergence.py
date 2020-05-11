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
from util import compute_divergence

def train_test_split(x, y, p):
    assert (len(x) == len(y))
    np.random.seed(1)
    msk = np.random.rand(len(x)) < p
    train_x, train_y = x[msk], y[msk]
    test_x, test_y = x[~msk], y[~msk]
    return train_x, train_y, test_x, test_y

# Load Data
pickle_file = 'datasets_parsed_2.pickle'
with open(pickle_file, "rb") as input_file:
    datasets = pickle.load(input_file)

filename = 'openml_divergence_log_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())+'.txt'
file = open(filename,'w')
file.write('Dataset\t Total Variation\n')
file.flush()

TV_train_list = []
TV_test_list = []

for i in range(85):
    # if i in [26, 37]:
    #     continue;
    X0, X1 = datasets[i]["X0"], datasets[i]["X1"]
    Y0, Y1 = datasets[i]["Y0"], datasets[i]["Y1"]

    # train/test split
    p = .7
    train_X0, train_Y0, test_X0, test_Y0 = train_test_split(X0, Y0, p)
    train_X1, train_Y1, test_X1, test_Y1 = train_test_split(X1, Y1, p)

    TV_train, TV_test = compute_divergence(train_X1, train_X0, test_X1, test_X0)
    if TV_train < 0:
        TV_train = 0.
    if TV_test < 0:
        TV_test = 0.

    file.write('{}\t {:.4f}\t {:.4f}\n'.format(i, TV_train, TV_test))
    file.flush()
    TV_train_list.append(TV_train)
    TV_test_list.append(TV_test)

pickle_save_file = filename+'.pickle'
f = open(pickle_save_file, 'wb')
save = {
    'TV_train': TV_train_list,
    'TV_test': TV_test_list
    }
pickle.dump(save, f, 2)
f.close()

file.close()
