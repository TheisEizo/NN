import numpy as np
from nn_v3_func import util

class layer:
    name = "Basic Layer"
    def __init__(self, ntype, *args, **kwargs):
        self.ntype = ntype()
    def __repr__(self): 
        return f'{self.name} with {self.ntype}'        
    def act(self,*args, **kwargs): 
        raise NotImplementedError(f'{self.name}')
    def diff(self, *args, **kwargs): 
        raise NotImplementedError(f'{self.name}')
    def init_ws_bs(self):
        raise NotImplementedError(f'{self.name}')
        
    def clean(self):
        del self.dws
        del self.dbs
        del self.ddws
        del self.ddbs
        
    def init_dws_dbs(self):
        self.dws = np.zeros(self.ws.shape)
        self.dbs = np.zeros(self.bs.shape)
        
    def update_dws_dbs(self):
        self.dws += self.ddws
        self.dbs += self.ddbs
        
    def update_ws_bs(self,batch, eta, n, reg, momentum):
        if momentum:
            self.vs = momentum*self.vs - eta/len(batch)*self.dws
            if reg:
                self.vs += -reg.act(eta=eta,ws=self.vs,n=n)
            self.ws += self.vs
        else:
            self.ws += -eta/len(batch)*self.dws
        self.bs += -eta/len(batch)*self.dbs
        if reg:
            self.ws += -reg.act(eta=eta,ws=self.ws,n=n)
            
class FullCon(layer):
    name = "Full Connected Layer"
    def __init__(self, ntype, wshape):
        self.ntype = ntype()
        self.init_ws_bs(wshape)
        
    def init_ws_bs(self, wshape):
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs = np.random.randn(1, wshape[1])

    def act(self, X, _):
        if len(X.shape) < 2: 
            X = X[np.newaxis,:]
        Z = np.dot(X, self.ws)+self.bs
        Y = self.ntype.act(Z)
        return Z, Y
    
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
    def __init__(self, ntype, p):
        self.ntype = ntype
        self.p = p
        self.init_ws_bs()
        
    def init_ws_bs(self):
        self.ws = np.array(0.)
        self.bs = np.array(0.)
        
    def act(self, X, train):
        if len(X.shape) < 2: 
            X = X[np.newaxis,:]
        if train:
            self.mask = np.ones(1)
        elif self.ntype == 'binomial':
            self.mask = np.random.binomial(1, self.p, size=X.shape)/self.p
        else:
            raise ValueError("Dropout ntype must be: 'binomial', ")
        Z = X
        Y = self.mask*Z
        return Z, Y
    
    def diff(self, da, X, l=None, Z=None):
        if l: 
            da = np.dot(da, l.ws.T)*self.mask.astype(float)
        del self.mask
        self.ddws = np.array(0.)
        self.ddbs = np.array(0.)
        return da

class Conv(layer):
    name = "Convelution Layer"
    def __init__(self, ntype, wshape, 
                 imshape=None, kshape=None, padding=(1, 0), roll=1):
        self.ntype = ntype()
        self.init_ws_bs(wshape)
        self.imshape = imshape
        self.kshape = kshape
        self.padding = padding #(width, value [number or 'min'])
        self.roll = roll
        
    def init_ws_bs(self, wshape):
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs = np.random.randn(1, wshape[1])
        
    def act(self, X, _):    
        if self.imshape == None: 
            self.imshape = (util.int_sqrt(X.shape[-1]),)*2
        if self.kshape == None: 
            self.kshape = (util.int_sqrt(self.ws.shape[0]),)*2
        X = X.reshape(X.shape[:-1]+self.imshape)
        X = self.im2col(X, self.padding, self.kshape, self.roll)
        X = np.dot(X, self.ws) + self.bs
        Z = X.reshape(X.shape[0],-1)
        Y = self.ntype.act(Z)
        return Z, Y
    
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
        self.init_ws_bs()
        self.imshape = imshape
        self.kshape = kshape
    
    def init_ws_bs(self):
        self.ws = np.array(0.)
        self.bs = np.array(0.)
        
    def act(self, X, _):
        if self.imshape[-1]==-1: 
            end_shape = (util.int_sqrt(X.shape[-1]/self.imshape[0]),)*2
            self.imshape = (self.imshape[0],) + end_shape
        X = X.reshape(X.shape[:-1]+self.imshape)
        Z = self.im_split(X, self.kshape)
        if self.ntype == 'max': 
            Y = np.amax(Z, axis=(-2,-1))
            self.pre_act = np.equal(Z, 
                                    np.amax(Z, axis=(-2,-1), keepdims=True))
        elif self.ntype == 'mean': 
            Y = np.mean(Z,axis=(-2,-1))
            self.pre_act = np.multiply(np.ones(Z.shape),
                                       np.mean(Z, axis=(-2,-1), keepdims=True))
        else:
            raise ValueError("Pool ntype must be: 'max', 'mean', ")
        Y = Y.reshape(Y.shape[0],-1)
        return Z, Y
    
    def diff(self, da, X, l=None, Z=None):
        if l: 
            da = np.dot(da, l.ws.T)
            da = self.im_collect(da, self.imshape, self.pre_act)
        del self.pre_act
        self.ddws = np.array(0.)
        self.ddbs = np.array(0.)
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
    def im_collect(X, imshape, modify=1.):
        X = X.reshape((X.shape[0],imshape[0],-1))
        X = X.reshape(X.shape+(1,1))*modify
        X = X.reshape((X.shape[0], -1))
        return X

class Networks(layer):
    name = 'Multiple Networks in Layer'

    def __init__(self, ntype, networks):
        self.ntype = ntype()
        self.networks = networks
        self.init_ws_bs(len(networks))
        
    def init_ws_bs(self, n):
        self.ws = np.random.randn(n, 1)/np.sqrt(n)
        self.bs = np.random.randn(1, 1)
        
    def train(self, train_data, test_data, 
            epochs, batch_size, 
            eta, reg, momentum, printout):
        for n, network in enumerate(self.networks):
            if printout: 
                print(f'\nNetwork layer {n+1}')
            network.SGD(train_data, test_data, 
                        epochs, batch_size, 
                        eta, reg, momentum, printout)
        if printout: 
            print('\nNetwork of networks')
    
    def act(self, X, _):
        A = self.act_layers(X)
        Z = np.dot(A, self.ws)+self.bs
        Y = self.ntype.act(np.squeeze(Z, axis=-1))   
        return Z, Y
    
    def diff(self, da, X, l=None, Z=None):
        X = self.act_layers(X)
        X = X.reshape((-1,X.shape[-1]))
        if l:
            da = np.dot(da, l.ws.T)*self.ntype.diff(Z)
            self.ddbs = np.sum(da, axis=0)
            self.ddws = np.dot(X.T, da)
        else:
            self.ddbs = np.sum(da); 
            self.ddws = np.dot(X.T,da.T)
        return da
    
    def act_layers(self, X):
        res = []
        for network in self.networks:
            network.act(X, train=False)
            res.append(network.cache[-1]['X'])
        Y = np.stack(res, axis=2)
        return Y

class RecurrentFullCon(layer):
    name = "Recurrent Full Connected Layer"
    
    def __repr__(self): 
        return f'{self.name} with {self.ntype_in} in and {self.ntype_out}' 
    
    def __init__(self, ntype_in, ntype_out, wshape, timesteps):
        self.ntype_in = ntype_in()
        self.ntype_out = ntype_out()
        self.init_ws_bs(wshape)
        self.timesteps = timesteps
        
    def init_ws_bs(self,wshape):
        self.ws_xh = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs_h  = np.random.randn(1, wshape[1])
        self.ws_hy = np.random.randn(wshape[1], wshape[2])/np.sqrt(wshape[1])
        self.bs_y  = np.random.randn(1, wshape[2])
        self.ws_hh = np.random.randn(wshape[1], wshape[1])/np.sqrt(wshape[1])
        
    def init_dws_dbs(self):
        self.dws_xh = np.zeros(self.ws_xh.shape)
        self.dbs_h = np.zeros(self.bs_h.shape)
        self.dws_hy = np.zeros(self.ws_hy.shape)
        self.dbs_y = np.zeros(self.bs_y.shape)
        self.dws_hh = np.zeros(self.ws_hh.shape)
    
    def init_ddws_ddbs(self):
        self.ddws_xh = np.zeros(self.ws_xh.shape)
        self.ddbs_h = np.zeros(self.bs_h.shape)
        self.ddws_hh = np.zeros(self.ws_hh.shape)
    
    def clean(self):
        del self.dws_xh
        del self.dbs_h
        del self.dws_hy
        del self.dbs_y
        del self.dws_hh
        del self.ddws_xh
        del self.ddbs_h
        del self.ddws_hy
        del self.ddbs_y
        del self.ddws_hh 
        
    def update_dws_dbs(self):
        self.dws_xh += self.ddws_xh
        self.dbs_h  += self.ddbs_h
        self.dws_hy += self.ddws_hy
        self.dbs_y  += self.ddbs_y
        self.dws_hh += self.ddws_hh
        
    def update_ws_bs(self,batch, eta, n, reg, momentum):
        if momentum:
            self.vs_xh = momentum*self.vs_xh - eta/len(batch)*self.dws_xh
            self.vs_hh = momentum*self.vs_hh - eta/len(batch)*self.dws_hh
            self.vs_hy = momentum*self.vs_hy - eta/len(batch)*self.dws_hy
            if reg:
                self.vs_xh += -reg.act(eta=eta,ws=self.vs_xh,n=n)
                self.vs_hh += -reg.act(eta=eta,ws=self.vs_hh,n=n)
                self.vs_hy += -reg.act(eta=eta,ws=self.vs_hy,n=n)
            self.ws_xh += self.vs_xh
            self.ws_hh += self.vs_hh
            self.ws_hy += self.vs_hy
        else:
            self.ws_xh += -eta/len(batch)*self.dws_xh
            self.ws_hh += -eta/len(batch)*self.dws_hh
            self.ws_hy += -eta/len(batch)*self.dws_hy
        self.bs_h += -eta/len(batch)*self.dbs_h
        self.bs_y += -eta/len(batch)*self.dbs_y
        if reg:
            self.ws += -reg.act(eta=eta,ws=self.ws,n=n)
    
    def act(self, X, train):
        if len(X.shape) < 2: 
            X = X[np.newaxis,:]
        X = X.reshape((X.shape[0], self.timesteps, -1))
        self.Hs = np.zeros((X.shape[0], self.timesteps+1, self.ws_hh.shape[1]))
        H = np.zeros(self.ws_hh.shape[0])
        self.Hs[:,0] = H
        
        for i in range(self.timesteps):
            H = np.dot(X[:,i], self.ws_xh)+np.dot(H, self.ws_hh)+self.bs_h
            H = self.ntype_in.act(H)
            self.Hs[:,i+1] = H
        Z = np.dot(H, self.ws_hy)+self.bs_y
        Y = self.ntype_out.act(Z)
        return Z, Y
    
    def diff(self, da, X, l=None, Z=None):
        self.init_ddws_ddbs()
        X = X.reshape((X.shape[0], self.timesteps, -1))

        self.ddws_hy = np.dot(self.Hs[:,-1].T, da)
        self.ddbs_y = da
        
        da_t = np.dot(da, self.ws_hy.T)
        
        for t in reversed(range(self.timesteps)):
            temp = self.ntype_in.diff(self.Hs[:,t+1])*da_t
            self.ddbs_h += temp
            self.ddws_hh += np.dot(temp,self.Hs[:,t].T)
            self.ddws_xh += np.dot(X[:,t].T,da_t)
            da_t = np.dot(da_t, self.ws_hh)
            
        for d in [self.ddws_xh, self.ddws_hh, self.ddws_hy, self.ddbs_h, self.ddbs_y]:
            np.clip(d, -1, 1, out=d)
            
        return da_t

#class BatchNorm(layer):
#https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py