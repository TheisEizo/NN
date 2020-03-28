import numpy as np

class reg:
    name = "Basic Regularization"    
    def __repr__(cls): return f'{cls.name}'
    @staticmethod
    def act(X): raise NotImplementedError

class L2(reg):
    def __init__(self, lmbda):
        self.lmbda = lmbda
    def act(self,**kwargs):
        return kwargs['eta']*(self.lmbda/kwargs['n'])*kwargs['ws']

class L1(reg):
    def __init__(self, lmbda):
        self.lmbda = lmbda
    def act(self,**kwargs):
        return kwargs['eta']*(self.lmbda/kwargs['n'])*np.sign(kwargs['ws'])  