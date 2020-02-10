import tensorflow as tf
import pickle
import gzip
import numpy as np
import scipy as sp
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from util import *
from sklearn.svm import LinearSVC

sns.set()
sns.set_style('white')

from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False



def label_Y(x, theta):
    '''
    labeling function
    :x: input feature value
    :theta: rotation angle
    '''
    assert len(x) == 2
    normal_vec = np.array([-np.sin(theta), np.cos(theta)]).reshape((2, 1))
    if x.dot(normal_vec) >= 0:
        y = 1
    else:
        y = 0
    return y


def gen_syn(loc, sigma, num_samp, theta):
    '''
    generate synthetic data
    :loc: Gaussian mean
    :sigma: Gaussian covariance matrix
    :num_samp: number of samples
    :theta: rotation angle
    '''
    
    assert len(loc) == len(sigma) and len(sigma) == len(sigma[0])
    assert num_samp > 0
    
    X = np.random.multivariate_normal(loc, sigma, int(num_samp))
    Y = np.zeros(num_samp)
    for i in range(num_samp):
        Y[i] = label_Y(X[i], theta)
    return X, Y

def bounds(theta):
    
    # Parameter Setting
    n_samples = 10000
    loc = np.array([0, 0])

    # covariance matrices
    sigma0 = np.array([[2, 0], [0, 1]])
    sigma1 = np.array([[2, 0.1], [0.1, 1]])
    
    # rotation angle for labeling functions
    theta0 = 0
    theta1 = theta
    
    # generate datasets for two populations
    X0_train, Y0_train = gen_syn(loc, sigma0, n_samples, theta0)
    X1_train, Y1_train = gen_syn(loc, sigma1, n_samples, theta1)
    X0_test, Y0_test = gen_syn(loc, sigma0, n_samples, theta0)
    X1_test, Y1_test = gen_syn(loc, sigma1, n_samples, theta1)
    
    # train SVM
    clf0 = LinearSVC(tol=1e-5)
    clf0.fit(X0_train, Y0_train)  
    
    clf1 = LinearSVC(tol=1e-5)
    clf1.fit(X1_train, Y1_train)  
    
    sigma0_inv = np.linalg.inv(sigma0)
    sigma1_inv = np.linalg.inv(sigma1)
    
    d_KL = 0.5 * (np.log(np.linalg.det(sigma1)/np.linalg.det(sigma0)) - 2.0 
                  + np.matrix.trace(sigma1_inv.dot(sigma0)))
    
    deno0 = np.sqrt(np.linalg.det((2.0 * sigma0_inv - sigma1_inv).dot(sigma1_inv)))
    assert deno0 > 0
    d_chisquare0 = np.linalg.det(sigma0_inv) / deno0 - 1.0
    
    deno1 = np.sqrt(np.linalg.det((2.0 * sigma1_inv - sigma0_inv).dot(sigma0_inv)))
    assert deno1 > 0
    d_chisquare1 = np.linalg.det(sigma1_inv) / deno1 - 1.0
    
    # VC dimension
    D = 3.0
    # with probability 1 - delta, the lower bound holds
    delta = 0.1
    assert delta > 0 and delta < 1.0
    
    nom = 2.0 * D * np.log(6.0 * n_samples) + 2.0 * np.log(16.0 / delta)
    
    # TODO update
    Omega_lb = 0
    #Omega_lb = 2.0 * 5.0 * np.sqrt(nom / n_samples)
    
    disc0 = np.sum(np.abs(clf1.predict(X0_test) - clf0.predict(X0_test))) / n_samples
    disc1 = np.sum(np.abs(clf1.predict(X1_test) - clf0.predict(X1_test))) / n_samples
    
    L0_hat = np.sum(np.abs(clf0.predict(X0_test) - Y0_test)) / n_samples
    L1_hat = np.sum(np.abs(clf1.predict(X1_test) - Y1_test)) / n_samples
    
    lambda_hat = 0.5 * (L0_hat + L1_hat)
    
    temp0 = (disc0 - lambda_hat - np.sqrt(d_KL/2)) / (np.sqrt(d_chisquare0 + 1.0) + 1.0)
    temp1 = (disc1 - lambda_hat - np.sqrt(d_KL/2)) / (np.sqrt(d_chisquare1 + 1.0) + 1.0)
    
    
    temp = max(temp0, temp1)
    if temp < 0:
        temp = 0
    lb = temp**2 - max(L0_hat, L1_hat) - Omega_lb
    
    if lb < 0:
        lb = 0
    
    Omega_ub= Omega_lb / 2.0
    ub= min(disc0, disc1) + Omega_ub
    
    if ub > 1.0:
        ub = 1.0
    
    return ub/2.0, lb*2.0



num_points = 10
theta = np.linspace(0, np.pi, num = num_points)
ub = []
true_val = []
lb = []
for i in range(num_points):
    ub_val, lb_val = bounds(theta[i])
    # TODO fill in true value
    true_val.append(theta[i] * 0.5 / np.pi)
    ub.append(ub_val)
    lb.append(lb_val)
    


plt.plot(theta, ub)
plt.plot(theta, true_val)
plt.plot(theta, lb)


plt.draw()






