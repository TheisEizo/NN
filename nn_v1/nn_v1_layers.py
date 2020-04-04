import numpy as np

class layer:
    name = "Basic Layer"
    def __init__(self, ntype, wshape):
        self.ntype = ntype()
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs = np.random.randn(1, wshape[1])
        self.dws = np.zeros(self.ws.shape)
        self.dbs = np.zeros(self.bs.shape)
        self.ddws = self.dws
        self.ddbs = self.dbs
        
    def __repr__(self): return f'{self.name} with {self.ntype}'        
    def act(X): raise NotImplementedError
    def diff(self, y): raise NotImplementedError  

class FullCon(layer):
    name = "Full Connected Layer"
    def act(self, X):
        if len(X.shape) < 2: X = X[np.newaxis,:]
        Z = np.dot(X, self.ws)+self.bs
        X = self.ntype.act(Z)
        return Z, X
    
    def diff(self, da, X, l=None, Z=None):
        if not l:
            self.ddbs = da; 
            self.ddws = np.dot(X.T, da)
        else:
            da = np.dot(da, l.ws.T) * self.ntype.diff(Z)
            self.ddbs = np.sum(da, axis=0)
            self.ddws = np.dot(X.T, da)
        return da

class Dropout(layer):
    name = "Dropout Layer"
    def __init__(self, p):
        self.p = p
        self.ntype = 'Binomial'
        self.mask = None
        
        nan = np.array(0.)
        self.ws = nan; self.bs = nan
        self.dws = nan; self.dbs = nan
        self.ddws = nan; self.ddbs = nan
        
    def act(self, X):
        if len(X.shape) < 2: X = X[np.newaxis,:]
        self.mask = np.random.binomial(1, self.p, size=X.shape) / self.p
        return X, self.mask*X
    
    def diff(self, da, X, l=None, Z=None):
        if not l: return da
        else: return np.dot(da, l.ws.T)*self.mask.astype(float)
        
