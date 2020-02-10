# -*- coding: utf-8 -*-
""""
Codes for ICML'20 on OpenML
Author: Hsiang Hsu
email:  hsianghsu@g.harvard.edu
"""
import pickle
import gzip
import numpy as np
from time import localtime, strftime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Load Data
pickle_file = 'datasets_parsed.pickle'
with open(pickle_file, "rb") as input_file:
    datasets = pickle.load(input_file)

filename = 'openml_log_'+strftime("%Y-%m-%d-%H.%M.%S", localtime())+'.txt'
file = open(filename,'w')

# train/test split
def train_test_split(x, y, p):
    assert (len(x) == len(y))
    np.random.seed(1)
    msk = np.random.rand(len(x)) < p
    train_x, train_y = x[msk], y[msk]
    test_x, test_y = x[~msk], y[~msk]
    return train_x, train_y, test_x, test_y

def epsilon_split(x0, y0, x1, y1, x0_test, y0_test, x1_test, y1_test):
    # number of samples
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    n = n0+n1

    # k-fold CV
    K = 5
    kf = KFold(n_splits=K)
    idx0 = kf.split(x0)
    idx1 = kf.split(x1)

    # hyperparameter to validate
    lambda_ = np.linspace(0, 1., 21)
    loss = np.zeros((len(lambda_), K))

    for j in range(K):
        train_idx0, val_idx0 = next(idx0)
        train_idx1, val_idx1 = next(idx1)
        x0_train, y0_train, x0_val, y0_val = x0[train_idx0], y0[train_idx0], x0[val_idx0], y0[val_idx0]
        x1_train, y1_train, x1_val, y1_val = x1[train_idx1], y1[train_idx1], x1[val_idx1], y1[val_idx1]


        x_train = np.concatenate((x0_train, x1_train), axis=0)
        y_train = np.concatenate((y0_train, y1_train), axis=0)


        for i in range(len(lambda_)):
            l = lambda_[i]
            weights = np.concatenate((l*n/n0*np.ones(x0_train.shape[0]), (1.-l)*n/n1*np.ones(x1_train.shape[0])))
            slvr = LogisticRegression(solver = 'liblinear', max_iter=10).fit(x_train, y_train, sample_weight = weights)
            loss[i, j] = max(np.abs(slvr.predict(x0_val) - y0_val).mean(), np.abs(slvr.predict(x1_val) - y1_val).mean())

    loss_mean = loss.mean(axis=1)
    opti_lambda = lambda_[np.argmin(loss_mean)]

    weights = np.concatenate((opti_lambda*n/n0*np.ones(x0_train.shape[0]), (1.-opti_lambda)*n/n1*np.ones(x1_train.shape[0])))
    slvr = LogisticRegression(solver = 'liblinear', max_iter=10).fit(x_train, y_train, sample_weight = weights)
    return max(np.abs(slvr.predict(x0_test) - y0_test).mean(), np.abs(slvr.predict(x1_test) - y1_test).mean())

def upper_lower(x0, y0, x1, y1, x0_test, y0_test, x1_test, y1_test):
    # number of samples
    n0 = x0.shape[0]
    n1 = x1.shape[0]
    n = n0+n1

    # solvers
    LR0 = LogisticRegression(solver = 'liblinear').fit(x0, y0)
    LR1 = LogisticRegression(solver = 'liblinear').fit(x1, y1)

    #
    disagreement0 = np.abs(LR0.predict(x0) - LR1.predict(x0)).mean()
    disagreement1 = np.abs(LR0.predict(x1) - LR1.predict(x1)).mean()
    print(disagreement0, disagreement1)
    upper = max(disagreement0, disagreement1)
    lower = min(disagreement0, disagreement1)

    #
    decoupling0_train = np.abs(LR0.predict(x0) - y0).mean()
    decoupling1_train = np.abs(LR1.predict(x1) - y1).mean()

    #
    decoupling0_test = np.abs(LR0.predict(x0_test) - y0_test).mean()
    decoupling1_test = np.abs(LR1.predict(x1_test) - y1_test).mean()
    return upper, lower, decoupling0_train, decoupling1_train, decoupling0_test, decoupling1_test

def compute_Omega(n0, n1, D, delta):
    Omega0 = np.sqrt((2*D+2*np.log(8/delta))/n0)
    Omega1 = np.sqrt((2*D+2*np.log(8/delta))/n1)
    return 4*max(Omega0, Omega1)


group_blind_loss_list = []
disagreement_max_list = []
disagreement_min_list = []
decoupling0_train_list = []
decoupling1_train_list = []
decoupling0_test_list = []
decoupling1_test_list = []
omega_list = []

for i in range(47):
    # Eliminate problematic datasets
    if i in [26, 37, 38]:
        continue;
    file.write('=== Dataset {} ===\n'.format(i))
    file.flush()

    X0, X1 = datasets[i]["X0"], datasets[i]["X1"]
    Y0, Y1 = datasets[i]["Y0"], datasets[i]["Y1"]

    # train/test split
    p = .7
    train_X0, train_Y0, test_X0, test_Y0 = train_test_split(X0, Y0, p)
    train_X1, train_Y1, test_X1, test_Y1 = train_test_split(X1, Y1, p)

    group_blind_loss = epsilon_split(train_X0, train_Y0, train_X1, train_Y1, test_X0, test_Y0, test_X1, test_Y1)
    disagreement_max, disagreement_min, decoupling0_train, decoupling1_train, decoupling0_test, decoupling1_test = upper_lower(train_X0, train_Y0, train_X1, train_Y1, test_X0, test_Y0, test_X1, test_Y1)

    file.write(' Group Blind Loss: {:.4f}\n'.format(group_blind_loss))
    file.write(' Disagreement\n  Max: {:.4f}\n  Min: {:.4f}\n'.format(disagreement_max, disagreement_min))
    file.write(' Decoupling train\n  Classifier0: {:.4f}\n  Classifier1: {:.4f}\n'.format(decoupling0_train, decoupling1_train))
    file.write(' Decoupling test\n  Classifier0: {:.4f}\n  Classifier1: {:.4f}\n'.format(decoupling0_test, decoupling1_test))
    file.flush()

    # Omega
    delta = .1
    n0, n1 = train_X0.shape[0], train_X1.shape[0]
    D = train_X0.shape[1] + 1
    omega = compute_Omega(n0, n1, D, delta)
    file.write(' Omega: {:.4f}\n'.format(omega))
    file.flush()

    group_blind_loss_list.append(group_blind_loss)
    disagreement_max_list.append(disagreement_max)
    disagreement_min_list.append(disagreement_min)
    decoupling0_train_list.append(decoupling0_train)
    decoupling1_train_list.append(decoupling1_train)
    decoupling0_test_list.append(decoupling0_test)
    decoupling1_test_list.append(decoupling1_test)
    omega_list.append(omega)

pickle_save_file = filename+'.pickle'
f = open(pickle_save_file, 'wb')
save = {
    'group_blind_loss': group_blind_loss_list,
    'disagreement_max': disagreement_max_list,
    'disagreement_min': disagreement_min_list,
    'decoupling0_train': decoupling0_train_list,
    'decoupling1_train': decoupling1_train_list,
    'decoupling0_test': decoupling0_test_list,
    'decoupling1_test': decoupling1_test_list,
    'omega': omega_list,
    }
pickle.dump(save, f, 2)
f.close()

file.close()
