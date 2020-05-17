import numpy as np
from func import util

class layer:
    name = "Basic Layer"
    def __init__(self, ntype, wshape):
        self.ntype = ntype()
        self.init_ws_bs(wshape)
        
    def __repr__(self): 
        return f'{self.name} with {self.ntype}'        
    def act(self,*args, **kwargs): 
        raise NotImplementedError(f'{self.name}')
    def diff(self, *args, **kwargs): 
        raise NotImplementedError(f'{self.name}')
    def init_ws_bs(self):
        raise NotImplementedError(f'{self.name}')
        
    def init_dws_dbs(self):
        self.dws = np.zeros(self.ws.shape)
        self.dbs = np.zeros(self.bs.shape)
        
    def init_vs(self):
        self.vs  = np.zeros(self.ws.shape)
        
    def clean(self):
        temps = [self.dws, self.dbs, self.ddws, self.ddbs]
        for temp in temps:
            del temp
        
    def clean_vs(self):
        del self.vs
        
    def update_dws_dbs(self):
        dgrads = [self.dws, self.dbs]
        ddgrads = [self.ddws, self.ddbs]
        for i, dgrad in enumerate(dgrads):
            dgrad += ddgrads[i]
        
    def update_ws_bs(self,batch, eta, n, reg, momentum):
        if momentum:
            self.vs = momentum*self.vs - eta/len(batch)*self.dws
            if reg:
                self.vs += -reg.act(eta=eta,ws=self.vs,n=n)
            self.ws += self.vs
        else:
            self.ws += -eta/len(batch)*self.dws
        if reg:
            self.ws += -reg.act(eta=eta,ws=self.ws,n=n)
        
        self.bs += -eta/len(batch)*self.dbs
    
class FullCon(layer):
    name = "Full Connected Layer"
        
    def init_ws_bs(self, wshape):
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs = np.random.randn(1, wshape[1])
    
    def clip_ddws_ddbs(self,ntype="minmax",gmin=-.25,gmax=.25):
        ddgrads = [self.ddws, self.ddbs]
        if ntype == 'minmax':
            for ddgrad in ddgrads:
                np.clip(ddgrad, gmin, gmax, out=ddgrad)
        elif ntype == 'norm':
            total_norm = 0
            for ddgrad in ddgrads:
                grad_norm = np.sum(np.power(ddgrad, 2))
                total_norm += grad_norm
            total_norm = np.sqrt(total_norm)
            clip_coef = gmax / (total_norm + 1e-6)
            if clip_coef < 1:
                for ddgrad in ddgrads:
                    ddgrad *= clip_coef
            
    def act(self, X, _):
        X = util.mindim(X)
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
        if self.ntype.name in ['ReLU Neuron',]:
            self.clip_ddws_ddbs()
        return da

class Dropout(layer):
    name = "Dropout Layer"
    def __init__(self, ntype, p):
        super().__init__(lambda: ntype, None)
        self.p = p
        
    def init_ws_bs(self, *args):
        self.ws = np.array(0.)
        self.bs = np.array(0.)
        
    def act(self, X, train):
        X = util.mindim(X)
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
        super().__init__(ntype, wshape)
        self.imshape = imshape
        self.kshape = kshape
        self.padding = padding
        self.roll = roll
        
    def init_ws_bs(self, wshape):
        self.ws = np.random.randn(wshape[0], wshape[1])/np.sqrt(wshape[0])
        self.bs = np.random.randn(1, wshape[1])
        
    def act(self, X, _):    
        if self.imshape == None: 
            self.imshape = (util.int_sqrt(X.shape[-1]),)*2
        if self.kshape == None: 
            self.kshape = (util.int_sqrt(self.ws.shape[0]),)*2
        X = util.mindim(X)
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
        super().__init__(lambda: ntype, None)
        self.imshape = imshape
        self.kshape = kshape
    
    def init_ws_bs(self, *args):
        self.ws = np.array(0.)
        self.bs = np.array(0.)
        
    def act(self, X, _):
        X = util.mindim(X)
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

#class BatchNorm(layer):
    #https://github.com/wiseodd/hipsternet/blob/master/hipsternet/layer.py

class Networks(layer):
    name = 'Multiple Networks in Layer'

    def __init__(self, ntype, networks):
        super().__init__(ntype, len(networks))
        self.networks = networks
        
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
        X = util.mindim(X)
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
            res.append(network.cache[-1]['Y'])
        Y = np.stack(res, axis=2)
        return Y

class RecurrentFullCon(layer):
    name = "Recurrent Full Connected Layer"

    def __init__(self, ntype, ntype_out, wshape):
        super().__init__(ntype, wshape)
        self.ntype_out = ntype_out()
        
    def __repr__(self): 
        return f'{self.name} with {self.ntype} in and {self.ntype_out}' 
        
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
        self.ddws_hy = np.zeros(self.ws_hy.shape)
        self.ddbs_y = np.zeros(self.bs_y.shape)
        self.ddws_hh = np.zeros(self.ws_hh.shape)
    
    def clean(self):
        temps = [self.dws_xh, self.dbs_h, self.dws_hy, self.dbs_y, 
                  self.dws_hh, self.ddws_xh, self.ddbs_h, self.ddws_hy, 
                  self.ddbs_y, self.ddws_hh,self.Hs, self.Ys]
        for temp in temps:
            del temp
                    
    def init_vs(self):
        self.vs_xh  = np.zeros(self.ws_xh.shape)
        self.vs_hh  = np.zeros(self.ws_hh.shape)
        self.vs_hy  = np.zeros(self.ws_hy.shape)
        
    def clean_vs(self):
        temps = [self.vs_xh, self.vs_hh, self.vs_hy]
        for temp in temps:
            del temp

    def update_dws_dbs(self):
        dgrads = [self.dws_xh, self.dbs_h, self.dws_hy, 
                      self.dbs_y, self.dws_hh]
        ddgrads = [self.ddws_xh, self.ddbs_h, self.ddws_hy, 
                      self.ddbs_y, self.ddws_hh]
        for dgrad, ddgrad in zip(dgrads, ddgrads):
            dgrad += ddgrad
    
    def clip_ddws_ddbs(self,ntype="norm",gmin=-1,gmax=1):
        ddgrads = [self.ddws_xh, self.ddbs_h, self.ddws_hy, 
                      self.ddbs_y, self.ddws_hh]
        if ntype == 'minmax':
            for ddgrad in ddgrads:
                np.clip(ddgrad, gmin, gmax, out=ddgrad)
        elif ntype == 'norm':
            total_norm = 0
            for ddgrad in ddgrads:
                grad_norm = np.sum(np.power(ddgrad, 2))
                total_norm += grad_norm
            total_norm = np.sqrt(total_norm)
            clip_coef = gmax / (total_norm + 1e-6)
            if clip_coef < 1:
                for ddgrad in ddgrads:
                    ddgrad *= clip_coef
            
    def update_ws_bs(self,batch, eta, n, reg, momentum):
        grads = [self.ws_xh, self.ws_hy, self.ws_hh]
        dgrads = [self.dws_xh, self.dws_hy, self.dws_hh]
        if momentum: 
            vgrads = [self.vs_xh, self.vs_hy, self.vs_hh]
        for i, grad in enumerate(grads):
            if momentum:
                vgrads[i] = momentum*vgrads[i] - eta/len(batch)*dgrads[i]
                
                if reg:
                    vgrads[i] += -reg.act(eta=eta,ws=vgrads[i],n=n)
                grad += vgrads[i]
            else:
                grad += - eta/len(batch)*dgrads[i]
            if reg:
                grads[i] += -reg.act(eta=eta,ws=grads[i],n=n)
        grads = [self.bs_h, self.bs_y]
        dgrads = [self.dbs_h, self.dbs_y]
        for i, grad in enumerate(grads):
            grad += - eta/len(batch)*dgrads[i]
        
    def act(self, X, train):
        if train : 
            self.Hs = [np.zeros((1,self.ws_hh.shape[0]))]
            self.Ys = []
            
            for t in range(len(X)):
                H_t = util.mindim(self.Hs[t].copy())
                X_t = util.mindim(X[t])
                H =  np.dot(X_t, self.ws_xh) + np.dot(H_t, self.ws_hh) + self.bs_h
                H =  self.ntype.act(H)
                Z =  np.dot(H, self.ws_hy)+self.bs_y
                Y =  self.ntype_out.act(Z)
                self.Hs.append(H)
                self.Ys.append(Y)
        else:
            Y = []
            for x in X[0]:
                _, y = self.act(x, train=True)
                Y.append(y)
            Y = np.array(Y)
        return 0, Y
    
    def diff(self, da, X, l=None, Z=None):
        self.init_ddws_ddbs()
        dda_h = np.zeros(self.Hs[0].shape)
        da_s = []
        for t in reversed(range(len(X))):
            da_t = util.mindim(da[t].copy())
            X_t = util.mindim(X[t])
            H_t = util.mindim(self.Hs[t].copy())
            H_t_prev = util.mindim(self.Hs[t-1].copy())
                
            self.ddws_hy += np.dot(H_t.T, da_t)
            self.ddbs_y += da_t
            
            da_h = dda_h + np.dot(da_t, self.ws_hy.T) 
            da_f = da_h * self.ntype.diff(H_t)
            da_s.append(da_f)
            self.ddws_xh += np.dot(X_t.T, da_f)
            
            self.ddws_hh += np.dot(H_t_prev.T, da_f)
            self.ddbs_h += da_f
            
            dda_h = np.dot(da_f, self.ws_hh.T)
            
        self.clip_ddws_ddbs('norm',gmax=0.25)
        return da_s

class LSTMFullCon(layer):
    name = "LSTM Full Connected Layer"

    def __init__(self, ntype, ntype_gate, ntype_out, wshape):
        super().__init__(ntype, wshape)
        self.ntype_gate = ntype_gate()
        self.ntype_out = ntype_out()
        
    def __repr__(self): 
        return (f'{self.name} with {self.ntype} in, {self.ntype_gate} gate'
                f'and {self.ntype_out}')
        
    def init_ws_bs(self,wshape):
        total_size = wshape[0]+wshape[1]
        self.ws_f = np.random.randn(total_size,wshape[1])/np.sqrt(total_size)
        self.bs_f = np.random.randn(1, wshape[1])
        self.ws_i = np.random.randn(total_size,wshape[1])/np.sqrt(total_size)
        self.bs_i = np.random.randn(1, wshape[1])
        self.ws_g = np.random.randn(total_size,wshape[1])/np.sqrt(total_size)
        self.bs_g = np.random.randn(1, wshape[1])
        self.ws_o = np.random.randn(total_size,wshape[1])/np.sqrt(total_size)
        self.bs_o = np.random.randn(1, wshape[1])
        self.ws_v = np.random.randn(wshape[1], wshape[2])/np.sqrt(wshape[1])
        self.bs_v  = np.random.randn(1, wshape[2])
        
    def init_dws_dbs(self):
        self.dws_f = np.zeros(self.ws_f.shape)
        self.dbs_f = np.zeros(self.bs_f.shape)
        self.dws_i = np.zeros(self.ws_i.shape)
        self.dbs_i = np.zeros(self.bs_i.shape)
        self.dws_g = np.zeros(self.ws_g.shape)
        self.dbs_g = np.zeros(self.bs_g.shape)
        self.dws_o = np.zeros(self.ws_o.shape)
        self.dbs_o = np.zeros(self.bs_o.shape)
        self.dws_v = np.zeros(self.ws_v.shape)
        self.dbs_v = np.zeros(self.bs_v.shape)
    
    def init_ddws_ddbs(self):
        self.ddws_f = np.zeros(self.ws_f.shape)
        self.ddbs_f = np.zeros(self.bs_f.shape)    
        self.ddws_i = np.zeros(self.ws_i.shape)
        self.ddbs_i = np.zeros(self.bs_i.shape)   
        self.ddws_g = np.zeros(self.ws_g.shape)
        self.ddbs_g = np.zeros(self.bs_g.shape)   
        self.ddws_o = np.zeros(self.ws_o.shape)
        self.ddbs_o = np.zeros(self.bs_o.shape)   
        self.ddws_v = np.zeros(self.ws_v.shape)
        self.ddbs_v = np.zeros(self.bs_v.shape)   
        
    def clean(self):
        temps = [self.dws_f, self.dbs_f, self.dws_i, self.dbs_i, 
                 self.dws_g, self.dbs_g, self.dws_o, self.dbs_o, 
                 self.dws_v, self.dbs_v, self.ddws_f, self.ddbs_f, 
                 self.ddws_i, self.ddbs_i, self.ddws_g, self.ddbs_g, 
                 self.ddws_o, self.ddbs_o, self.ddws_v, self.ddbs_v,
                 self.Zs, self.Fs, self.Is, self.Gs, 
                 self.Os, self.Cs, self.Hs]
        for temp in temps:
            del temp
                    
    def init_vs(self):
        self.vs_f  = np.zeros(self.ws_f.shape)
        self.vs_i  = np.zeros(self.ws_i.shape)
        self.vs_g  = np.zeros(self.ws_g.shape)
        self.vs_o  = np.zeros(self.ws_o.shape)
        self.vs_v  = np.zeros(self.ws_v.shape)
        
    def clean_vs(self):
        temps = [self.vs_f, self.vs_i, self.vs_g, self.vs_o, self.vs_v]
        for temp in temps:
            del temp

    def update_dws_dbs(self):
        dgrads = [self.dws_f, self.dbs_f, self.dws_i, self.dbs_i, 
                 self.dws_g, self.dbs_g, self.dws_o, self.dbs_o, 
                 self.dws_v, self.dbs_v]
        ddgrads = [self.ddws_f, self.ddbs_f, self.ddws_i, self.ddbs_i, 
                   self.ddws_g, self.ddbs_g, self.ddws_o, self.ddbs_o, 
                   self.ddws_v, self.ddbs_v]
        for dgrad, ddgrad in zip(dgrads, ddgrads):
            dgrad += ddgrad
    
    def clip_ddws_ddbs(self,ntype,gmin=-1,gmax=1):
        ddgrads = [self.ddws_f, self.ddbs_f, self.ddws_i, self.ddbs_i, 
                   self.ddws_g, self.ddbs_g, self.ddws_o, self.ddbs_o, 
                   self.ddws_v, self.ddbs_v]
        if ntype == 'minmax':
            for ddgrad in ddgrads:
                np.clip(ddgrad, gmin, gmax, out=ddgrad)
        elif ntype == 'norm':
            total_norm = 0
            for ddgrad in ddgrads:
                grad_norm = np.sum(np.power(ddgrad, 2))
                total_norm += grad_norm
            total_norm = np.sqrt(total_norm)
            clip_coef = gmax / (total_norm + 1e-6)
            if clip_coef < 1:
                for ddgrad in ddgrads:
                    ddgrad *= clip_coef
            
    def update_ws_bs(self,batch, eta, n, reg, momentum):
        grads = [self.ws_f, self.ws_i,self.ws_g, self.ws_o, self.ws_v]
        dgrads = [self.dws_f, self.dws_i,self.dws_g, self.dws_o, self.dws_v]
        if momentum: 
            vgrads = [self.vs_f, self.vs_i, self.vs_g, self.vs_o, self.vs_v]
        for i, grad in enumerate(grads):
            if momentum:
                vgrads[i] = momentum*vgrads[i] - eta/len(batch)*dgrads[i]
                
                if reg:
                    vgrads[i] += -reg.act(eta=eta,ws=vgrads[i],n=n)
                grad += vgrads[i]
            else:
                grad += - eta/len(batch)*dgrads[i]
            if reg:
                grads[i] += -reg.act(eta=eta,ws=grads[i],n=n)
        grads = [self.bs_f, self.bs_i,self.bs_g, self.bs_o, self.bs_v]
        dgrads = [self.dbs_f, self.dbs_i,self.dbs_g, self.dbs_o, self.dbs_v]
        for i, grad in enumerate(grads):
            grad += - eta/len(batch)*dgrads[i]
    
    def act(self, X, train):
        
        if train: 
            self.Zs = []
            self.Fs = []
            self.Is = []
            self.Gs = []
            self.Os = []
            self.Cs = [np.zeros(self.bs_f.shape)]
            self.Hs = [np.zeros(self.bs_f.shape)]

            for t in range(len(X)):
                X_t = util.mindim(X[t])
                H_t = util.mindim(self.Hs[t])
                C_t = util.mindim(self.Cs[t])
                
                Z = np.hstack((H_t, X_t))
                self.Zs.append(Z)
                
                F = self.ntype_gate.act(np.dot(Z, self.ws_f)+self.bs_f)
                self.Fs.append(F)
                
                I = self.ntype_gate.act(np.dot(Z, self.ws_i)+self.bs_i)
                self.Is.append(I)
                
                G = self.ntype_gate.act(np.dot(Z, self.ws_g)+self.bs_g)
                self.Gs.append(G)
                
                C_t = F*C_t + I*G
                self.Cs.append(C_t)
                
                O = self.ntype_gate.act(np.dot(Z, self.ws_o)+self.bs_o)
                self.Os.append(O)
                
                H_t = O*self.ntype.act(C_t)
                self.Hs.append(H_t)
                
            V = np.dot(H_t, self.ws_v)+self.bs_v
            Y = self.ntype_out.act(V)

        else:
            Y = []
            for x in X[0]:
                _, y = self.act(x, train=True)
                Y.append(y)
            Y = np.array(Y)
        return 0, Y
    
    def diff(self, da, X, l=None, Z=None):
        self.init_ddws_ddbs()
        
        C_t_next = np.zeros_like(self.Cs[0])
        H_t_next = np.zeros_like(self.Hs[0])
        
        da_s = []
        for t in reversed(range(len(X))):
            O_t = self.Os[t]
            I_t = self.Is[t]
            G_t = self.Gs[t]
            F_t = self.Fs[t]
            Z_t = self.Zs[t]
            H_t = self.Hs[t]
            C_t = self.Cs[t]

            C_t_prev = self.Cs[t-1]
            da_t = util.mindim(da[t])
            
            self.ddws_v += np.dot(H_t.T, da_t)
            self.ddbs_v += da_t
            
            da_Hs = H_t_next.copy()
            da_Hs += np.dot(da_t, self.ws_v.T)
            da_Os = da_Hs*self.ntype.act(C_t)
            da_Os = da_Os*self.ntype_gate.diff(O_t)
            self.ddws_o += np.dot(Z_t.T,da_Os)
            self.ddbs_o += da_Os          
            
            da_Cs = C_t_next.copy()
            da_Cs += da_Hs*O_t*self.ntype.diff(C_t)
            da_Gs = da_Cs * I_t
            da_Gs = self.ntype.diff(G_t)*da_Gs
            self.ddws_g += np.dot(Z_t.T, da_Gs)
            self.ddbs_g += da_Gs
            
            da_s.append(da_Cs)
            
            da_Is = da_Cs * G_t
            da_Is = self.ntype_gate.act(I_t)*da_Is
            self.ddws_i += np.dot(Z_t.T, da_Is)
            self.ddbs_i += da_Is
            
            da_Fs = da_Cs * C_t_prev
            da_Fs = self.ntype_gate.act(F_t)*da_Fs
            self.ddws_f += np.dot(Z_t.T,da_Fs)
            self.ddbs_f += da_Fs
            
        self.clip_ddws_ddbs('norm',gmax=0.25)
        return da_s

class GRUFullCon(layer):
    name = "GRU Full Con Layer"

    def __init__(self, ntype, ntype_gate, ntype_out, wshape):
        super().__init__(ntype, wshape)
        self.ntype_gate = ntype_gate()
        self.ntype_out = ntype_out()
        
    def __repr__(self): 
        return (f'{self.name} with {self.ntype} in, {self.ntype_gate} gate'
                f'and {self.ntype_out}')
        
    def init_ws_bs(self,wshape):
        total_size = wshape[0]+wshape[1]
        self.ws_z = np.random.randn(total_size,wshape[1])/np.sqrt(total_size)
        self.bs_z = np.random.randn(1, wshape[1])
        self.ws_r = np.random.randn(total_size,wshape[1])/np.sqrt(total_size)
        self.bs_r = np.random.randn(1, wshape[1])
        self.ws_g = np.random.randn(total_size,wshape[1])/np.sqrt(total_size)
        self.bs_g = np.random.randn(1, wshape[1])
        self.ws_v = np.random.randn(wshape[1], wshape[2])/np.sqrt(wshape[1])
        self.bs_v  = np.random.randn(1, wshape[2])
        
    def init_dws_dbs(self):
        self.dws_z = np.zeros(self.ws_z.shape)
        self.dbs_z = np.zeros(self.bs_z.shape)
        self.dws_r = np.zeros(self.ws_r.shape)
        self.dbs_r = np.zeros(self.bs_r.shape)
        self.dws_g = np.zeros(self.ws_g.shape)
        self.dbs_g = np.zeros(self.bs_g.shape)
        self.dws_v = np.zeros(self.ws_v.shape)
        self.dbs_v = np.zeros(self.bs_v.shape)
    
    def init_ddws_ddbs(self):
        self.ddws_z = np.zeros(self.ws_z.shape)
        self.ddbs_z = np.zeros(self.bs_z.shape)    
        self.ddws_r = np.zeros(self.ws_r.shape)
        self.ddbs_r = np.zeros(self.bs_r.shape)   
        self.ddws_g = np.zeros(self.ws_g.shape)
        self.ddbs_g = np.zeros(self.bs_g.shape)   
        self.ddws_v = np.zeros(self.ws_v.shape)
        self.ddbs_v = np.zeros(self.bs_v.shape)   
        
    def clean(self):
        temps = [self.dws_z, self.dbs_z, self.dws_r, self.dbs_r, 
                 self.dws_g, self.dbs_g, self.dws_v, self.dbs_v, 
                 self.ddws_z, self.ddbs_z, self.ddws_r, self.ddbs_r, 
                 self.ddws_g, self.ddbs_g, self.ddws_v, self.ddbs_v,
                 ]
        for temp in temps:
            del temp
                    
    def init_vs(self):
        self.vs_z  = np.zeros(self.ws_z.shape)
        self.vs_r  = np.zeros(self.ws_r.shape)
        self.vs_g  = np.zeros(self.ws_g.shape)
        self.vs_v  = np.zeros(self.ws_v.shape)
        
    def clean_vs(self):
        temps = [self.vs_z, self.vs_r, self.vs_g, self.vs_v]
        for temp in temps:
            del temp

    def update_dws_dbs(self):
        dgrads = [self.dws_z, self.dbs_z, self.dws_r, self.dbs_r, 
                 self.dws_g, self.dbs_g, self.dws_v, self.dbs_v]
        ddgrads = [self.ddws_z, self.ddbs_z, self.ddws_r, self.ddbs_r, 
                   self.ddws_g, self.ddbs_g, self.ddws_v, self.ddbs_v]
        for dgrad, ddgrad in zip(dgrads, ddgrads):
            dgrad += ddgrad
    
    def clip_ddws_ddbs(self,ntype,gmin=-1,gmax=1):
        ddgrads = [self.ddws_z, self.ddbs_z, self.ddws_r, self.ddbs_r, 
                   self.ddws_g, self.ddbs_g, self.ddws_v, self.ddbs_v]
        if ntype == 'minmax':
            for ddgrad in ddgrads:
                np.clip(ddgrad, gmin, gmax, out=ddgrad)
        elif ntype == 'norm':
            total_norm = 0
            for ddgrad in ddgrads:
                grad_norm = np.sum(np.power(ddgrad, 2))
                total_norm += grad_norm
            total_norm = np.sqrt(total_norm)
            clip_coef = gmax / (total_norm + 1e-6)
            if clip_coef < 1:
                for ddgrad in ddgrads:
                    ddgrad *= clip_coef
            
    def update_ws_bs(self,batch, eta, n, reg, momentum):
        grads = [self.ws_z, self.ws_r,self.ws_g, self.ws_v]
        dgrads = [self.dws_z, self.dws_r,self.dws_g, self.dws_v]
        if momentum: 
            vgrads = [self.vs_z, self.vs_r, self.vs_g, self.vs_v]
        for i, grad in enumerate(grads):
            if momentum:
                vgrads[i] = momentum*vgrads[i] - eta/len(batch)*dgrads[i]
                
                if reg:
                    vgrads[i] += -reg.act(eta=eta,ws=vgrads[i],n=n)
                grad += vgrads[i]
            else:
                grad += - eta/len(batch)*dgrads[i]
            if reg:
                grads[i] += -reg.act(eta=eta,ws=grads[i],n=n)
        grads = [self.bs_z, self.bs_r, self.bs_g, self.bs_v]
        dgrads = [self.dbs_z, self.dbs_r, self.dbs_g, self.dbs_v]
        for i, grad in enumerate(grads):
            grad += - eta/len(batch)*dgrads[i]
    
    def act(self, X, train):
        if train: 
            self.Xs = []
            self.Zs = []
            self.Rs = []
            self.Gs = []
            self.Hs = [np.zeros(self.bs_z.shape)]


            for t in range(len(X)):
                X_ti = util.mindim(X[t])
                H_t = util.mindim(self.Hs[t])
                X_t = np.hstack((H_t, X_ti))
                self.Xs.append(X_t)
                
                Z = self.ntype_gate.act(np.dot(X_t, self.ws_z)+self.bs_z)
                self.Zs.append(Z)
                
                R = self.ntype_gate.act(np.dot(X_t, self.ws_r)+self.bs_r)
                self.Rs.append(R)
                
                X_t = np.hstack((H_t*R, X_ti))
                G = self.ntype.act(np.dot(X_t, self.ws_g)+self.bs_g)
                self.Gs.append(G)
                
                H_t = (1-Z)*H_t+Z*G
                self.Hs.append(H_t)
                
            V = np.dot(H_t, self.ws_v)+self.bs_v
            Y = self.ntype_out.act(V)

        else:
            Y = []
            for x in X[0]:
                x = util.mindim(x)
                _, y = self.act(x, train=True)
                Y.append(y)
            Y = np.array(Y)
        return 0, Y
    
    def diff(self, da, X, l=None, Z=None):
        self.init_ddws_ddbs()
        H_t_next = np.zeros_like(self.Hs[0])
        
        da_s = []
        for t in reversed(range(len(X))):
            X_t = self.Xs[t]
            Z_t = self.Zs[t]
            R_t = self.Rs[t]
            G_t = self.Gs[t]
            H_t = self.Hs[t]
            
            H_t_prev = self.Hs[t-1]
            da_t = util.mindim(da[t])
            
            self.ddws_v += np.dot(H_t.T, da_t)
            self.ddbs_v += da_t
            
            da_Hs = H_t_next.copy()
            da_Hs += np.dot(da_t, self.ws_v.T)
            
            da_Gs = da_Hs*Z_t
            da_Gs = da_Gs*self.ntype.diff(G_t)
            self.ddws_g += np.dot(X_t.T,da_Gs)
            self.ddbs_g += da_Gs       
            
            da_Rs = H_t_prev*da_Gs
            da_Rs = self.ntype_gate.diff(R_t)*da_Rs
            self.ddws_r += np.dot(X_t.T,da_Rs)
            self.ddbs_r += da_Gs 
            
            da_Zs = self.ntype_gate.diff(Z_t)
            self.ddws_z += np.dot(X_t.T, da_Zs)
            self.ddbs_z += da_Zs
            
            da_s.append(da_Hs)
            
        self.clip_ddws_ddbs('norm',gmax=0.25)
        return da_s