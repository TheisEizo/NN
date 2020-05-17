import numpy as np
from func import util

class nn:
    name = "Basic Neural Network"
    def __init__(self, layers, costf):
        self.layers = layers
        self.costf = costf()
        
    def __repr__(self):
        return f'{self.name} with \n{self.layers} and \n{self.costf}'
        
    def SGD(self, train_data, test_data=None, 
            epochs=10, batch_size=10, 
            eta=0.1, reg=None, momentum=None,
            printout="Loss",
            ):
        
        for l in self.layers:
            if l.name == 'Multiple Networks in Layer':
                l.train(train_data, None, 
                      epochs, batch_size, 
                      eta, reg, momentum, 
                      printout)
        
        X, y = train_data        
        if test_data: 
            X_t, y_t = test_data            
        n = len(X)
        init_lst = np.arange(n)
        if momentum:
            for l in self.layers:
                l.init_vs()
        for i in range(epochs):
            np.random.shuffle(init_lst)
            X, y = X[init_lst], y[init_lst]
            batches = [(X[k:k+batch_size], y[k:k+batch_size]) 
                        for k in np.arange(0, n, batch_size)]
            for batch in batches: 
                self.update(batch, eta, n, reg, momentum)
            if printout=="Accuracy":
                print(f'Epoch {i+1} out of {epochs}')
                print('\tAccuracy on training data: '+str(self.accuracy(X, y)))
                if test_data: 
                    print('\tAccuracy on test data: '+str(self.accuracy(X_t, y_t)))
            if printout=="Loss":
                print(f'Epoch {i+1} out of {epochs}')
                print('\tLoss on training data: '+str(self.loss(X, y)))
                if test_data: 
                    print('\tLoss on test data: '+str(self.loss(X_t, y_t)))
        del self.cache
        for l in self.layers:
            l.clean()
            if momentum:
                l.clean_vs()
            
    def update(self, batch, eta, n, reg, momentum):
        for l in self.layers:
            l.init_dws_dbs()
        for X, y in zip(batch[0], batch[1]):
            self.act(X, train=True)
            _ = self.diff(y)
            for l in self.layers:
                l.update_dws_dbs()
        for l in self.layers:
            l.update_ws_bs(batch, eta, n, reg, momentum)
        del self.cache
    
    def loss(self, X, y):
        self.act(X, train=False)
        loss = self.costf.act(self.cache[-1]['Z'], self.cache[-1]['Y'], y)
        return loss

    def predict(self, X):
        self.act(X, train=False)
        res = self.cache[-1]['Y']
        return res
    
    def accuracy(self, X, y):
        #This if-loop is a memory fix
        if (len(y) <= 1000) or (len(y)/1000 != len(y)//1000):
            y_pred = self.predict(X)
            res = (y_pred.argmax(axis=-1) == y.argmax(axis=-1))
        else: 
            res = []           
            for Xi, yi in zip(np.split(X, 50), np.split(y, 50),):
                yi_pred = self.predict(Xi)
                pred = (yi_pred.argmax(axis=-1) == yi.argmax(axis=-1))
                res.append(pred)
        mean_pred = np.mean(res)
        return mean_pred
    
    def act(X): 
        raise NotImplementedError
        
    def diff(self, y): 
        raise NotImplementedError
        
class FF(nn):
    name = "Feed Forward Neural Network"
    
    def act(self, X, train):
        X = util.mindim(X)
        self.cache = [{'Z':None, 'Y':X}]
        for l in self.layers:
            Z, X = l.act(X, train)
            self.cache.append({'Z':Z,'Y':X})

    def diff(self, y):        
        for n in range(-1,-len(self.layers)-1,-1):
            l = self.layers[n]
            if n == -1: 
                da = self.costf.diff(self.cache[n]['Z'], 
                                     self.cache[n]['Y'], 
                                     y)
                da = l.diff(da, self.cache[n-1]['Y'])
            else:
                da = l.diff(da, 
                            self.cache[n-1]['Y'], 
                            self.layers[n+1], 
                            self.cache[n]['Z'])
        return da
    
class GAN(nn):
    name = "Generative Adversarial Network"
    def __init__(self, gen, dis, costf):
        self.gen = gen
        self.dis = dis
        self.costf = costf()
    
    def SGD(self, train_data, 
            epochs=10, batch_size=10, 
            eta=0.1, eta_decay = 0.,
            reg=None, momentum=None,
            printout="Loss", printoutImage=False,
            ):
        
        for l in self.gen.layers+self.dis.layers:
            if l.name == 'Multiple Networks in Layer':
                l.train(train_data, None, 
                      epochs, batch_size, 
                      eta, reg, momentum, 
                      printout)
        
        X, _ = train_data                   
        n = len(X)
        init_lst = np.arange(n)
        if momentum:
            for l in self.gen.layers+self.dis.layers:
                l.init_vs()
        if printoutImage:
                import matplotlib.pyplot as plt
        for i in range(epochs):
            np.random.shuffle(init_lst)
            X = X[init_lst]
            batches = [(X[k:k+batch_size]) 
                        for k in np.arange(0, n, batch_size)]
            for batch in batches: 
                self.update(batch, eta, n, reg, momentum)
            if printout=="Loss":
                loss = self.loss(X)
                print(f'Epoch {i+1} out of {epochs}')
                print('\tGen loss on training data: '+str(loss[0]))
                print('\tDis loss on training data: '+str(loss[1]))
            if printoutImage:
                z = self.generate(1)
                plt.imshow(z.reshape(28,28))
                plt.title(f'Epoch {i+1}')
                plt.show()
            eta *= (1-eta_decay)
            print(f'Learn Rate= {eta}')
        
        del self.gen.cache
        del self.dis.cache
        del self.dis.cache_true
        del self.dis.cache_fake
        for l in self.gen.layers+self.dis.layers:
            l.clean()
            if momentum:
                l.clean_vs()
                
    def update(self, batch, eta, n, reg, momentum):
        for l in self.gen.layers+self.dis.layers:
            l.init_dws_dbs()
            
        for X_true in batch:
        #if True:
            #X_true = batch
            #print(X_true.shape)
            self.act(X_true, train=True)
            self.diff(None)
            
            for l in self.gen.layers+self.dis.layers:
                l.clip_ddws_ddbs()
                l.update_dws_dbs()
                
        for l in self.gen.layers+self.dis.layers:
            l.update_ws_bs(batch, eta, n, reg, momentum)
    
    def generate(self, n):
        z_len = self.gen.layers[0].ws.shape[0]
        z = np.random.normal(0, 1, (n, z_len))
        self.gen.act(z, train=True)
        X_fake = self.gen.cache[-1]['Y']
        return X_fake
    
    def discriminate(self, X, train=False):
        self.dis.act(X, train)
        return self.dis.cache
    
    def act(self, X, train=False):
        X_true = util.mindim(X)
        X_fake = self.generate(len(X_true))
        self.dis.cache_true = self.discriminate(X_true, train)
        self.dis.cache_fake = self.discriminate(X_fake, train)
        
    def diff(self, _):
        self.dis.cache = self.dis.cache_true
        _ = self.dis.diff(self.dis.cache_true[-1]['Y'])
        true_dis_ddws_ddbs = [(l.ddws, l.ddbs) for l in self.dis.layers]
        
        self.dis.cache = self.dis.cache_fake
        _ = self.dis.diff(None)
        fake_dis_ddws_ddbs = [(l.ddws, l.ddbs) for l in self.dis.layers]
        
        da = self.dis.diff(self.dis.cache_fake[-1]['Y'])
        da = np.dot(da, self.dis.layers[0].ws.T)
        _ = self.gen.diff((da,))
        
        zips = zip(self.dis.layers, true_dis_ddws_ddbs,fake_dis_ddws_ddbs)
        for l, (t_w, t_b),(f_w, f_b) in zips:
            l.ddws = t_w + f_w
            l.ddbs = t_b + f_b

    def loss(self, X):
        self.act(X, train=False)
        
        Gloss = self.costf.act(None, self.dis.cache_fake[-1]['Y'], None)
        Dloss = self.costf.act(None, self.dis.cache_fake[-1]['Y'], 
                               self.dis.cache_true[-1]['Y'])
        return (Gloss, Dloss)