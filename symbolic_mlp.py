#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  4 16:34:18 2020

@author: dora
"""

import numpy as np
import casadi as cas

class neuron:
    def __init__(self, n_in, activation='tanh'):
        self.setActivationFunction(activation)
        self.activation = activation
        assert n_in >= 1
        self.number_in = n_in
        self.weights = cas.SX.sym('w', self.number_in)

    def setActivationFunction(self, str):
        if str in 'tanh':
            self.activation_function = self.act_tanh
        elif str in 'logistic':
            self.activation_function = self.act_logistic
        elif str in 'relu':
            self.activation_function = self.act_relu
        elif str in 'softmax':
            self.activation_function = self.act_softmax
        else:
            raise ValueError('Could not find given activation function.')

    def out(self, inn):
        inn = cas.vec(inn)
        assert self.number_in == inn.size()[0], 'Wrong number of inputs given.'
        if self.activation == 'softmax':
            y = self.activation_function(self.weights*inn)
        else:
            y = self.activation_function(cas.dot(self.weights, inn))
        return y

    @staticmethod
    def act_tanh(u):
        return cas.tanh(u)
    @staticmethod
    def act_logistic(u):
        return 1./(1 + cas.exp(-u))
    @staticmethod
    def act_relu(u):
        return cas.fmax(0,u)
    @staticmethod
    def act_softmax(u):
        return cas.exp(u) / cas.sum1(cas.exp(u))

def indicator_array(nvals, idx):
    ind = np.zeros(nvals)
    ind[idx] = 1.
    return ind

def xlogy(x, y):
    return cas.if_else(x>0, cas.if_else(y>0, x*cas.log(y), -1e16), 0)

# properties matchin sklearn's MLPClassifier (https://scikit-learn.org/stable/modules/neural_networks_supervised.html)
class MLPClassifier:
    def __init__(self, neurons, activation='relu', random_state=0, alpha=1e-6):
        assert len(neurons) > 2, 'An MLP must consist of at least 3 layers.'
        assert not np.any(np.asarray(neurons) < 1), 'All numbers of neurons per layer must be a positive number.'

        self.activation = activation
        self.random_state = random_state
        self.alpha = alpha
        self.number_neurons = neurons
        self.number_layers = len(neurons)

        self.sym = { 'u': [], 'w':[], 'y': [], 'fcn':[] }
        self.sym['u'] = cas.SX.sym('u', self.number_neurons[0])
        ytmp = self.sym['u']
        self.sym['w'] = []

        rg = np.random.RandomState(self.random_state)
        self.weights = []

        for l in range(1,self.number_layers):
            utmp = ytmp
            ytmp = []

            if l < self.number_layers-1:
                # hidden layers
                factor = 6. # for initialization
                bias = cas.SX.sym('bias', self.number_neurons[l])
                self.sym['w'] = cas.vertcat(self.sym['w'], bias) # bias before weights, according to sklearn's implementation
                for n in range(self.number_neurons[l]):
                    neu = neuron(np.shape(utmp)[0], self.activation)
                    ytmp = cas.vertcat(ytmp, neu.out(utmp) + bias[n])
                    self.sym['w'] = cas.vertcat(self.sym['w'], neu.weights)
            else:
                # output layer: softmax, no bias
                factor = 2. # for initialization
                for n in range(self.number_neurons[l]):
                    neu = neuron(np.shape(utmp)[0], 'softmax')
                    ytmp = cas.vertcat(ytmp, neu.out(utmp))
                    self.sym['w'] = cas.vertcat(self.sym['w'], neu.weights)

            # initialization from sklearn's MLPClassifier class
            init_bound = np.sqrt(factor / (self.number_neurons[l-1] + self.number_neurons[l]))
            w_init = cas.vec(rg.uniform(-init_bound, init_bound, (self.number_neurons[l-1], self.number_neurons[l])))
            if l < self.number_layers-1:
                # add bias initializations for hidden layers
                w_init = cas.vertcat(cas.vec(rg.uniform(-init_bound, init_bound, (self.number_neurons[l]))), cas.vec(w_init))

            self.weights = cas.vertcat(self.weights, cas.vec(w_init))

        self.sym['y'] = ytmp
        self.sym['fcn'] = cas.Function('fcn', [self.sym['u'], self.sym['w']], [self.sym['y']], ['u', 'w'], ['yhat'])
        self.number_weights = self.sym['w'].size()[0]

    def fit(self, input, output):
        nsamples = input.shape[0]

        fobj = 0
        for n in range(nsamples):
            yhat = self.sym['fcn'].call({'u': input[n,:], 'w': self.sym['w']})['yhat']
            y = indicator_array(yhat.shape, output[n])
            fobj -= cas.sum1(xlogy(y, yhat) + xlogy((1-y), 1-yhat)) # cross entropy loss

        fobj = fobj/nsamples # scaling
        if self.alpha:
            fobj += self.alpha*cas.dot(self.sym['w'],self.sym['w']) # L2 regularization

        nlp = { 'x': self.sym['w'], 'f': fobj, 'g': [] }

        opts = {'print_time': False, 'ipopt': { 'tol': 1e-3, 'max_iter': 1000, 'hessian_approximation': 'limited-memory', 'print_level': 0 } }
        solver = cas.nlpsol('nlpsol', 'ipopt', nlp, opts)

        res = solver(x0=self.weights)
        self.weights = np.asarray(res['x'])

    def predict(self, input):
        nsamples = input.shape[0]
        res = []
        for n in range(nsamples):
            y_bin = self.sym['fcn'].call({'u': input[n,:], 'w': self.weights})['yhat'] # predicted probabilities

            # find index of significant value
            r = np.ravel(y_bin > 0.5).nonzero()[0] # 0.5 threshold
            if (r.size == 0) or (r.size > 1):
                r = np.asarray(-1) # inconclusive results are mapped to -1
            res.append(r)

        return np.hstack(res).astype(int)

    def score(self, input, output):
        ypred = self.predict(input)
        return np.average(ypred == output)
