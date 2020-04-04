import numpy as np

class cost:
    name = "Basic Cost Function"   
    def __repr__(cls): return f'{cls.name}'
    def act(y_pred, y_true): raise NotImplementedError
    def diff(y_pred, y_true): raise NotImplementedError 

class SquaredLoss(cost):
    name = "Squared Loss Cost Function"
    @staticmethod
    def act(Z, X, y): return 0.5 * np.linalg.norm(X - y)**2 
    @staticmethod
    def diff(Z, X, y): return (X - y) * 1/(1+np.exp(-Z)) * (1 - 1/(1+np.exp(-Z)))

class CrossEntropy(cost):
    name = "Cross Entropy Cost Function"
    @staticmethod
    def act(Z, X, y): return np.sum(np.nan_to_num(-y*np.log(X)-(1-y)*np.log(1-X)))
    @staticmethod
    def diff(Z, X, y): return (X - y)