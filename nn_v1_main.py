# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:45:06 2020

@author: Theis Eizo
"""
#%%
import numpy as np

import os
os.chdir('C:/Users/45414/Python')

from nn_v4_func import import_MNIST
(X_train, y_train), (X_test, y_test) = import_MNIST()

#import scipy as sp
#import matplotlib.pyplot as plt
#%%
class util:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(np.argmax(y_pred,axis=1) == np.argmax(y_true,axis=1))
    @staticmethod
    def onehot(y):
        y = np.array(y, int)
        res = np.zeros([y.size, np.max(y) + 1])
        res[range(y.size), y] = 1.
        return res

class neuron:
    name = "Basic Neuron"    
    def __repr__(cls):
        return f'{cls.name}'
    @staticmethod
    def act(X):
        raise NotImplementedError
    @staticmethod
    def diff(X):
        raise NotImplementedError  
        
class Sigmoid(neuron):
    name = 'Sigmoid Neuron'
    @staticmethod
    def act(X):
        return 1/(1+np.exp(-X))
    @staticmethod
    def diff(X):
        return Sigmoid.act(X)*(1-Sigmoid.act(X))
    
class layer:
    name = "Basic Layer"
    def __init__(self, ntype, wshape):
        self.ntype = ntype()
        self.ws = self.init_layer(wshape)
    def __repr__(self):
        return f'{self.name} with {self.ntype}'
        
    @staticmethod
    def init_layer(wshape):
        return np.random.randn(wshape[0]+1, wshape[1])/np.sqrt(wshape[1])

    def act(X):
        raise NotImplementedError
    def diff(self, cache, y):
        raise NotImplementedError  

class FullCon(layer):
    name = "Full Connected Layer"
    def act(self, X):
        if len(X.shape) < 2: X = X[np.newaxis,:]
        Z = np.hstack((np.ones((X.shape[0],1)), X)).dot(self.ws)
        X = self.ntype.act(Z)
        return Z, X
    
class nn:
    name = "Basic Neural Net"
    def __init__(self, layers, costf):
        self.layers = layers
        self.costf = costf()
        
    def __repr__(self):
        try:
            return f'{self.name} with {self.layers} and {self.costf}'
        except AttributeError:
            return f'{self.name}'
        
    def SGD(self, X, y, epochs=2, batch_size=12, eta=0.01, test_data = None):
        n = len(X)
        y = util.onehot(y)
        for i in range(epochs):
            init_lst = np.arange(n)
            np.random.shuffle(init_lst)
            X, y = X[init_lst], y[init_lst]
            batches = [(X[k:k+batch_size], y[k:k+batch_size]) 
                        for k in range(0, n, batch_size)]
            for batch in batches:
                self.update(batch, eta)
            print(f'Epoch {i+1} out of {epochs}')
            print('Accuracy on training data: '+str((util.accuracy(y, self.act(X)[-1]['X']))))
            if test_data: 
                X_t, y_t = test_data
                y_t = util.onehot(y_t)
                print('Accuracy on test data: '+str((util.accuracy(y_t, self.act(X_t)[-1]['X']))))
            
    def update(self, batch, eta):
        nabla = [np.zeros(l.ws.shape) for l in self.layers]
        for X, y in zip(batch[0], batch[1]):
            cache = self.act(X)
            delta_nabla = self.diff(cache, y)
            nabla = [n+dn for n, dn in zip(nabla, delta_nabla)]
        for n,l in enumerate(self.layers):
            l.ws = l.ws-(eta/len(batch))*nabla[n]

    def act(X):
        raise NotImplementedError
    def diff(self, cache, y):
        raise NotImplementedError  
        
class FF(nn):
    name = "Feed Forward Network"
    
    def act(self, X):
        if len(X.shape) < 2: X = X[np.newaxis,:]
        cache = [{'Z':None, 'X':X}]
        for l in self.layers:
            Z, X = l.act(X)
            cache.append({'Z':Z,'X':X})
        return cache
    
    def diff(self, cache, y):
        ln = len(cache)
        nabla = [np.zeros(l.ws.shape) for l in self.layers]
        delta = self.costf.diff(cache[-1]['X'], y)
        nabla[-1] = delta.dot(cache[-2]['X'].T)
        for n,l in enumerate(reversed(self.layers[:-1])):
            sp = l.ntype.diff(cache[1-n-ln]['X'])
            delta = l.ws.dot(delta.T)*sp
            nabla[-n-2] = cache[-n-ln]['X'].dot(delta[1:])
        return nabla
    
class cost:
    name = "Basic Cost Function"   
    def __repr__(cls):
        return f'{cls.name}'
    def act(y_pred, y_true):
        raise NotImplementedError
    def diff(y_pred, y_true):
        raise NotImplementedError 

class SquaredLoss(cost):
    name = "Squared Loss Cost Function" 
    def act(y_pred, y_true):
        return 0.5 * np.sum((y_true - y_pred)**2) / y_pred.shape[0]
    @staticmethod
    def diff(y_pred, y_true):
        return (y_pred - y_true) / y_pred.shape[0]
    
NN = FF([
        FullCon(Sigmoid, (28*28,10)),
        FullCon(Sigmoid, (10,10)),
        ], 
    SquaredLoss)