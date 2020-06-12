import numpy as np
from typing import Type

class neuron:
    """
    Base neuron class, not to be used as activation neuron for network
    layers
    """
    name = "Basic Neuron"    
    def __repr__(cls) -> str: 
        return f'{cls.name}'
    @staticmethod
    def act(X: np.ndarray) -> np.ndarray: 
        raise NotImplementedError
    @staticmethod                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            
    def diff(X: np.ndarray) -> np.ndarray: 
        raise NotImplementedError
                                                                                        
class Sigmoid(neuron):                                                                                                                                                                              
    name = 'Sigmoid Neuron'
    @staticmethod
    def act(X: np.ndarray) -> np.ndarray:  
        return 1/(1+np.exp(-X))
    @staticmethod
    def diff(X: np.ndarray) -> np.ndarray: 
        sigX = Sigmoid.act(X)
        return sigX*(1-sigX)
    
class ReLU(neuron):
    name = 'ReLU Neuron'
    @staticmethod
    def act(X: np.ndarray, a: float=1e-6) -> np.ndarray:
        return np.maximum(a, X)
    @staticmethod
    def diff(X: np.ndarray, a: float=1e-6) -> np.ndarray:
        return np.where(X>=0.0, 1.0, a)

class LReLU(neuron):
    name = 'Leaky ReLU Neuron'
    @staticmethod
    def act(X: np.ndarray, a: float=1e-3) -> np.ndarray:
        return np.maximum(a*X, X)
    @staticmethod
    def diff(X: np.ndarray, a: float=1e-3) -> np.ndarray:
        return np.where(X>0.0, 1.0, a)

class Tanh(neuron):
    name = 'Tanh Neuron'
    @staticmethod
    def act(X: np.ndarray) -> np.ndarray: 
        return np.tanh(X)
    @staticmethod
    def diff(X: np.ndarray) -> np.ndarray: 
        return 1/np.power(np.cosh(X),2)
    
class Softmax(neuron):
    name = 'Softmax Neuron'
    @staticmethod
    def act(X: np.ndarray) -> np.ndarray: 
        expX = np.exp(X - X.max())
        return expX / np.sum(expX,axis=1)[:,None]
    @staticmethod
    def diff(X: np.ndarray) -> np.ndarray: 
        smX = Softmax.act(X)
        return smX*(1-np.sum(smX))

class Identity(neuron):
    name = 'Kernel Neuron'
    @staticmethod
    def act(X: np.ndarray) -> np.ndarray: 
        return X
    @staticmethod
    def diff(X: np.ndarray) -> np.ndarray: 
        return np.ones(X.shape)

class Swish(neuron):
    name = 'Swish Neuron'
    def __init__(self, b: float) -> None:
        self.b = b
    def __call__(self) -> Type[neuron]:
        return self
    def act(self, X: np.ndarray) -> np.ndarray: 
        return np.multiply(X,Sigmoid.act(self.b * X))
    def diff(self, X: np.ndarray) -> np.ndarray: 
        bsigX = Sigmoid.act(self.b*X)
        bswiX = self.b*self.act(X)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        return bswiX + np.multiply(bsigX,(1-bswiX))