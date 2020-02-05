#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:33:02 2020

@author: dora
"""

"""This script compares an sklearn MLP classifier with a symbolic implementation."""

from sklearn.neural_network import MLPClassifier as MLPClassifier_sklearn
import numpy as np
from utils import fetch_data
from symbolic_mlp import MLPClassifier as MLPClassifier_symbolic

# VARIABLES
NFIT = 1000 # number of digits to fit MLP
NGRID = 10 # number of columns/rows that the test result will show
NTEST = NGRID**2 # number of samples of images used for testing

# PREPARATION
data = fetch_data(NFIT, NTEST)

# SKLEARN
sklearn_mlp = MLPClassifier_sklearn(solver='lbfgs', activation='relu', alpha=1e-4, hidden_layer_sizes=(10,), random_state=2019, verbose=0)
sklearn_mlp.fit(data['train']['images_scaled'], data['train']['target'])
sklearn_pred = sklearn_mlp.predict(data['test']['images_scaled'])

# SYMBOLIC
symbolic_mlp = MLPClassifier_symbolic([64, 10, 1], activation='relu', alpha=1e-4, random_state=2019)
symbolic_mlp.fit(data['train']['images_scaled'], data['train']['target'])
symbolic_pred = symbolic_mlp.predict(data['test']['images_scaled'])

# OVERVIEW
print(
    "Scores on test data",
    "\n\tsklearn-mlp:", sklearn_mlp.score(data['test']['images_scaled'], data['test']['target']),
    "\n\tsymbolic-mlp:", symbolic_mlp.score(data['test']['images_scaled'], data['test']['target'])
)

import matplotlib.pyplot as plt
import matplotlib.gridspec as gs

plt.figure()
grid = gs.GridSpec(NGRID, NGRID)
grid.update(wspace=0, hspace=1)
for i in range(NGRID**2):
    ax = plt.subplot(grid[i])
    plt.axis('off')
    ax.set_aspect('equal')
    ax.imshow(data['test']['images'][i], cmap=plt.cm.gray_r, interpolation='nearest')
    ax.text(-4, 0, data['test']['target'][i], fontsize=8)
    col_sk = (0, 1, 0) if sklearn_pred[i] == data['test']['target'][i] else (1, 0, 0)
    ax.text(-4, 4, sklearn_pred[i], fontsize=8, color=col_sk)
    col_sy = (0, 1, 0) if symbolic_pred[i] == data['test']['target'][i] else (1, 0, 0)
    ax.text(-4, 8, symbolic_pred[i], fontsize=8, color=col_sy)

plt.show()
