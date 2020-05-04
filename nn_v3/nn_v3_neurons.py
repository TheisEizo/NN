import numpy as np

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
        sigX = 1/(1+np.exp(-X))
        return sigX*(1-sigX)
    
class ReLU(neuron):
    name = 'ReLU Neuron'
    @staticmethod
    def act(X, a=0.0): 
        return np.maximum(a*X, X)
    @staticmethod
    def diff(X, a=0.0): 
        return np.where(X>=0.0, 1.0, a)

class LReLU(neuron):
    name = 'Leaky ReLU Neuron'
    @staticmethod
    def act(X, a=1e-3): 
        return np.maximum(a*X, X)
    @staticmethod
    def diff(X, a=1e-3): 
        return np.where(X>0.0, 1.0, a)

class Tanh(neuron):
    name = 'Tanh Neuron'
    @staticmethod
    def act(X): 
        return np.tanh(X)
    @staticmethod
    def diff(X): 
        return 1/np.power(np.cosh(X),2)
    
class Softmax(neuron):
    name = 'Softmax Neuron'
    @staticmethod
    def act(X):
        expX = np.exp(X - X.max())
        return expX / np.sum(expX,axis=1)[:,None]
    @staticmethod
    def diff(X):
        smX = Softmax.act(X)
        return smX*(1-np.sum(smX))

class Identity(neuron):
    name = 'Kernel Neuron'
    @staticmethod
    def act(X): 
        return X
    @staticmethod
    def diff(X): 
        return np.ones(X.shape)
    