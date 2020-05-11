import numpy as np

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
    @staticmethod
    def act(Z, X, y): 
        return 0.5 * np.linalg.norm(X - y)**2
    @staticmethod
    def diff(Z, X, y): 
        sigX = 1/(1+np.exp(-Z))
        return (X - y) * sigX * (1 - sigX)

class CrossEntropy(cost):
    name = "Cross Entropy Cost Function"
    @staticmethod
    def act(Z, X, y): 
        return np.sum(np.nan_to_num(-y*np.log(X)-(1-y)*np.log(1-X)))

    @staticmethod
    def diff(Z, X, y): 
        return (X - y)

class RNNCrossEntropy(cost):
    name = "RNN Cross Entropy Cost Function"
    @staticmethod
    def act(Z, X, y): 
        loss = 0
        for X_i, y_i in zip(X,y):
            loss += -np.mean(np.nan_to_num(np.log(X_i+1e-12)*y_i))
        return loss
    @staticmethod
    def diff(Z, X, y): 
        return (X - y)