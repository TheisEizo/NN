import numpy as np

class layer:
    name = "Basic Layer"
    def __init__(self, ntype, wshape):
        self.ntype = ntype()
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[1])
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

class Dropout(layer): ##FIX
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
        if not l: 
            pass
        else: 
            da = np.dot(da, l.ws.T)
            return da*self.mask.astype(float)
    
class BatchNorm(layer): ##FIX
    def exp_running_avg(running, new, gamma=.9):
        return gamma * running + (1. - gamma) * new
    
    def bn_forward(self, X, gamma, beta, cache, momentum=.9, train=True):
        running_mean, running_var = cache
        eps = 1e-8
        if train:
            mu = np.mean(X, axis=0)
            var = np.var(X, axis=0)
    
            X_norm = (X - mu) / np.sqrt(var + eps)
            out = gamma * X_norm + beta
    
            cache = (X, X_norm, mu, var, gamma, beta)
    
            running_mean = self.exp_running_avg(running_mean, mu, momentum)
            running_var = self.exp_running_avg(running_var, var, momentum)
        else:
            X_norm = (X - running_mean) / np.sqrt(running_var + eps)
            out = gamma * X_norm + beta
            cache = None
    
        return out, cache, running_mean, running_var

    def bn_backward(dout, cache):
        eps = 1e-8
        X, X_norm, mu, var, gamma, beta = cache
    
        N, D = X.shape
    
        X_mu = X - mu
        std_inv = 1. / np.sqrt(var + eps)
    
        dX_norm = dout * gamma
        dvar = np.sum(dX_norm * X_mu, axis=0) * -.5 * std_inv**3
        dmu = np.sum(dX_norm * -std_inv, axis=0) + dvar * np.mean(-2. * X_mu, axis=0)
    
        dX = (dX_norm * std_inv) + (dvar * 2 * X_mu / N) + (dmu / N)
        dgamma = np.sum(dout * X_norm, axis=0)
        dbeta = np.sum(dout, axis=0)
    
        return dX, dgamma, dbeta
    

    