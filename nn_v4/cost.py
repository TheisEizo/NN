import numpy as np

class cost:
    name = "Basic Cost Function"   
    def __repr__(cls): 
        return f'{cls.name}'
    def act(_, y_pred, y_true): 
        raise NotImplementedError
    def diff(_, y_pred, y_true): 
        raise NotImplementedError 

class SquaredLoss(cost):
    name = "Squared Loss Cost Function"
    @staticmethod
    def act(_, X, y): 
        return 0.5 * np.linalg.norm(X - y)**2
    @staticmethod
    def diff(Z, X, y): 
        sigX = 1/(1+np.exp(-Z))
        return (X - y) * sigX * (1 - sigX)

class CrossEntropy(cost):
    name = "Cross Entropy Cost Function"
    @staticmethod
    def act(_, X, y): 
        return np.sum(np.nan_to_num(-y*np.log(X)-(1-y)*np.log(1-X)))

    @staticmethod
    def diff(_, X, y): 
        return (X - y)

class RNNCrossEntropy(cost):
    name = "RNN Cross Entropy Cost Function"
    @staticmethod
    def act(_, X, y): 
        e = 1e-12
        loss = 0
        for X_i, y_i in zip(X,y):
            loss += -np.mean(np.nan_to_num(np.log(X_i+e)*y_i))
        return loss
    @staticmethod
    def diff(_, X, y): 
        return (X - y)

class GANCrossEntropy(cost):
    name = "GAN Cross Entropy Cost Function"
    
    @staticmethod
    def act(_, X_fake, X_real=None): 
        e = 1e-12
        if type(X_real) == np.ndarray:
            return np.mean(-np.log(X_real+e) - np.log(1 - X_fake+e))
        else:
            return np.mean(-np.log(X_fake+e))
    
    @staticmethod
    def diff(_, X_fake, X_real=None): 
        e = 1e-12
        if type(X_real) == np.ndarray:
            return -1/(X_real+e)
        elif type(X_real) == tuple:
            return X_real[0]
        else:
            return 1/(1 - X_fake+e)