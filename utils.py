#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:35:36 2020

@author: dora
"""

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
import numpy as np

def fetch_data(nfit, ntest):
    digits = datasets.load_digits()

    ndata = len(digits.data)
    X_train = digits['images'][:nfit]
    X_test = digits['images'][-ntest:]

    scaler = StandardScaler()
    scaler.fit(X_train.reshape(nfit, 64))
    X_train_scaled = scaler.transform(X_train.reshape(nfit, 64))
    X_test_scaled = scaler.transform(X_test.reshape(ntest, 64))

    data = {}
    data['train'] = { 'images': X_train, 'images_scaled': X_train_scaled, 'target': digits['target'][:nfit] }
    data['test'] = { 'images': X_test, 'images_scaled': X_test_scaled, 'target': digits['target'][-ntest:] }

    return data
