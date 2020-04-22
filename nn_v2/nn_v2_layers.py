import numpy as np
from nn_v2_func import util

class layer:
    name = "Basic Layer"
    def __init__(self, ntype, *args, **kwargs):
        self.ntype = ntype()
        
    def __repr__(self): return f'{self.name} with {self.ntype}'        
    def act(self,*args, **kwargs): raise NotImplementedError(f'{self.name}')
    def diff(self, *args, **kwargs): raise NotImplementedError(f'{self.name}')  

class FullCon(layer):
    name = "Full Connected Layer"
    def __init__(self, ntype, wshape):
        self.ntype = ntype()
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs = np.random.randn(1, wshape[1])
        self.dws = np.zeros(self.ws.shape)
        self.dbs = np.zeros(self.bs.shape)
        self.ddws = self.dws
        self.ddbs = self.dbs
    
    def act(self, X):
        if len(X.shape) < 2: 
            X = X[np.newaxis,:]
        Z = np.dot(X, self.ws)+self.bs
        X = self.ntype.act(Z)
        return Z, X
    
    def diff(self, da, X, l=None, Z=None):
        if l:
            da = np.dot(da, l.ws.T) * self.ntype.diff(Z)
            self.ddbs = np.sum(da, axis=0)
            self.ddws = np.dot(X.T, da)
        else:
            self.ddbs = da; 
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
        if len(X.shape) < 2: 
            X = X[np.newaxis,:]
        self.mask = np.random.binomial(1, self.p, size=X.shape)/self.p
        Z = self.mask*X
        return X, Z
    
    def diff(self, da, X, l=None, Z=None):
        if l: 
            da = np.dot(da, l.ws.T)*self.mask.astype(float)
        return da

class Conv(layer):
    name = "Convelution Layer"
    def __init__(self, ntype, wshape, imshape=None, kshape=None, padding=(1, 0), roll=1):
        self.ntype = ntype()
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs = np.random.randn(1, wshape[1])
        self.dws = np.zeros(self.ws.shape)
        self.dbs = np.zeros(self.bs.shape)
        self.ddws = self.dws
        self.ddbs = self.dbs
        
        self.imshape = imshape
        self.padding = padding #Tuple with (Padding width, paddig value (float or 'min'))
        self.kshape = kshape
        self.roll = roll

    def act(self, X):    
        if self.imshape == None: self.imshape = (util.int_sqrt(X.shape[-1]),)*2
        if self.kshape == None: self.kshape = (util.int_sqrt(self.ws.shape[0]),)*2
        X = X.reshape(X.shape[:-1]+self.imshape)
        X = self.im2col(X, self.padding, self.kshape, self.roll)
        X = np.dot(X, self.ws) + self.bs
        Z = X.reshape(X.shape[0],-1)
        X = self.ntype.act(Z)
        return Z, X
    
    def diff(self, da, X, l=None, Z=None):
        if l:
            da = da.reshape((da.shape[0], -1, self.ws.shape[-1]))
            Z = Z.reshape((Z.shape[0], -1, self.ws.shape[-1]))
            da = np.dot(da, l.ws.T) * self.ntype.diff(Z)
            
            X = X.reshape(X.shape[:-1]+self.imshape)
            X = self.im2col(X, self.padding, self.kshape, self.roll)
            da = da.reshape((-1, da.shape[-1]))
            X = X.reshape((-1, X.shape[-1]))     
            self.ddbs = np.sum(da, axis=0)
            self.ddws = np.dot(X.T, da)
        else:
            self.ddbs = da; 
            self.ddws = np.dot(X.T, da)
        return da
    
    @staticmethod
    def im2col(X, padding, kshape, roll):
        p_shape = ((0,0),)*(len(X.shape)-2)+((padding[0],)*2,)*2
        if padding[1] == 'min': 
            X = np.pad(X, p_shape, 'constant', constant_values=X.min())
        else:
            X = np.pad(X, p_shape, 'constant', constant_values=padding[1])
        
        X = np.array([np.roll(X, -i*roll, axis=-1)[:,:,:kshape[-1]] 
                    for i in range(0,X.shape[-1]-kshape[-1]+1,roll)])
        X = np.array([np.roll(X, -i*roll, axis=-2)[:,:,:kshape[-2]] 
                    for i in range(0,X.shape[-2]-kshape[-2]+1,roll)])
        
        X = X.reshape(X.shape[:-2]+(-1,))
        X = X.reshape((-1,)+X.shape[-2:])
        X = np.moveaxis(X,-2,0)
        return X
    
class Pool(layer):
    name = "Pooling Layer"
    def __init__(self, ntype, imshape, kshape):
        self.ntype = ntype
        self.imshape = imshape
        self.kshape = kshape
        self.pre_act = None
        
        nan = np.array(0.)
        self.ws = nan; self.bs = nan
        self.dws = nan; self.dbs = nan
        self.ddws = nan; self.ddbs = nan
        
    def act(self, X):
        if self.imshape[-1]==-1: 
            end_shape = (util.int_sqrt(X.shape[-1]/self.imshape[0]),)*2
            self.imshape = (self.imshape[0],) + end_shape
        X = X.reshape(X.shape[:-1]+self.imshape)
        Z = self.im_split(X, self.kshape)
        if self.ntype == 'max': 
            X = np.amax(Z, axis=(-2,-1))
            self.pre_act = np.equal(Z, np.amax(Z, axis=(-2,-1), keepdims=True))
        if self.ntype == 'mean': 
            X = np.mean(Z,axis=(-2,-1))
            self.pre_act = np.ones(Z.shape)*np.mean(Z, axis=(-2,-1), keepdims=True)
        X = X.reshape(X.shape[0],-1)
        return Z, X
    
    def diff(self, da, X, l=None, Z=None):
        if l: 
            da = np.dot(da, l.ws.T)
            da = self.im_collectAndModify(da, self.imshape, self.pre_act)
        return da
        
    @staticmethod    
    def im_split(X, kshape):
        X = np.array(np.split(X,X.shape[-1]/kshape[-1],axis=-1))
        X = np.array(np.split(X,X.shape[-2]/kshape[-2],axis=-2))
        X = np.moveaxis(X,2,0)
        X = X.reshape((X.shape[0],)+(-1,)+X.shape[3:])
        X = np.moveaxis(X,1,2)
        return X
    
    @staticmethod
    def im_collectAndModify(X, imshape, modifyer):
        X = X.reshape((X.shape[0],imshape[0],-1))
        X = X.reshape(X.shape+(1,1))*modifyer
        X = X.reshape((X.shape[0], -1))
        return X
    
class BatchNorm(layer): ##FIX THIS
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
        
