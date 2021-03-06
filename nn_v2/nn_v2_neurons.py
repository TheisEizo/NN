import numpy as np

class neuron:
    name = "Basic Neuron"    
    def __repr__(cls): return f'{cls.name}'
    @staticmethod
    def act(X): raise NotImplementedError
    @staticmethod
    def diff(X): raise NotImplementedError  
        
class Sigmoid(neuron):
    name = 'Sigmoid Neuron'
    @staticmethod
    def act(X): return 1/(1+np.exp(-X))
    @classmethod
    def diff(cls, X): 
        sigX = cls.act(X)
        return sigX*(1-sigX)
    
class ReLU(neuron):
    name = 'ReLU Neuron'
    @staticmethod
    def act(X): return np.maximum(0,X)
    @staticmethod
    def diff(X): return np.where(X>0, 1, 0)

class LReLU(neuron):
    name = 'Leaky ReLU Neuron'
    @staticmethod
    def act(X, a=1e-3): return np.maximum(a*X, X)
    @staticmethod
    def diff(X, a=1e-3): return np.where(X>0, 1, a)

class Tanh(neuron):
    name = 'Tanh Neuron'
    @staticmethod
    def act(X): return np.tanh(X)
    @classmethod
    def diff(cls, X): return 1/np.power(np.cosh(X),2)
    
class Softmax(neuron):
    name = 'Softmax Neuron'
    @staticmethod
    def act(X):
        expX = np.exp(X - X.max())
        return expX / np.sum(expX,axis=1)[:,None]
    @classmethod
    def diff(cls, X):
        smX = cls.act(X)
        return smX*(1-np.sum(smX))

class Identity(neuron):
    name = 'Kernel Neuron'
    @staticmethod
    def act(X): return X
    @staticmethod
    def diff(X): return np.ones(X.shape)
    