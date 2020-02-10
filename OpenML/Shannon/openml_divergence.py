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

# Load Data
pickle_file = 'datasets_parsed.pickle'
with open(pickle_file, "rb") as input_file:
    datasets = pickle.load(input_file)

file = open('openml_divergence_log.txt','w')
file.write('Total Variation\n')
file.flush()

for i in range(47):
    X0, X1 = datasets[i]["X0"], datasets[i]["X1"]
    TV = compute_divergence(X1, X0)
    file.write('Dataset {}\t: {}\n'.format(i, TV))
    file.flush()

file.close()
