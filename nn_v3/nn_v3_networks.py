from nn_v3_func import util
import numpy as np

class nn:
    name = "Basic Neural Network"
    def __init__(self, layers, costf):
        self.layers = layers
        self.costf = costf()
        self.cache = None
        
    def __repr__(self):
        return f'{self.name} with {self.layers} and {self.costf}'
        
    def SGD(self, train_data, test_data=None, 
            epochs=10, batch_size=10, 
            eta=0.1, reg=None, momentum=None,
            ):
        
        for l in self.layers:
            if l.name == 'Multiple Networks in Layer':
                l.train(train_data, None, 
                      epochs, batch_size, 
                      eta, reg, momentum)
        
        X, y = train_data
        y = util.onehot(y)
        if test_data: 
            X_t, y_t = test_data
            y_t = util.onehot(y_t)
            
        n = len(X)
        init_lst = np.arange(n)
        for i in range(epochs):
            np.random.shuffle(init_lst)
            X, y = X[init_lst], y[init_lst]
            batches = [(X[k:k+batch_size], y[k:k+batch_size]) 
                        for k in np.arange(0, n, batch_size)]
            for batch in batches: 
                self.update(batch, eta, n, reg, momentum)
            print(f'Epoch {i+1} out of {epochs}')
            print('\tAccuracy on training data: '+str(self.accuracy(X, y)))
            if test_data: 
                print('\tAccuracy on test data: '+str(self.accuracy(X_t, y_t)))
        
    def update(self, batch, eta, n, reg, momentum):
        for l in self.layers:
            l.dbs = np.zeros(l.bs.shape); l.dws = np.zeros(l.ws.shape)
        
        for X, y in zip(batch[0], batch[1]):
            self.act(X, train=True)
            self.diff(y)
            del self.cache
            for l in self.layers:
                l.dbs += l.ddbs; l.dws += l.ddws
        for l in self.layers:
            if reg:
                l.ws -= reg.act(eta=eta,ws=l.ws,n=n)
            if momentum:
                if not 'vs' in dir(l): l.vs = np.zeros(l.ws.shape)
                l.vs = momentum*l.vs - eta/len(batch)*l.dws
                l.ws += l.vs
            else:
                l.ws -= eta/len(batch)*l.dws
            l.bs -= eta/len(batch)*l.dbs 
        
    def predict(self, X):
        self.act(X)
        res = self.cache[-1]['X'].argmax(axis=-1)
        del self.cache
        return res
        
    def act(X): 
        raise NotImplementedError
    def diff(self, y): 
        raise NotImplementedError
    
    def accuracy(self, X, y):
        if (len(y) <= 1000) or (len(y)/1000 != len(y)//1000):
            self.act(X, train=False)
            y_pred = self.cache[-1]['X']
            res = y_pred.argmax(axis=-1) == y.argmax(axis=-1)
            del self.cache
        else: #MEMORY FIX
            res = []           
            for Xi, yi in zip(np.split(X, 50), np.split(y, 50),):
                self.act(Xi, train=False)
                yi_pred = self.cache[-1]['X']
                del self.cache
                pred = yi_pred.argmax(axis=-1) == yi.argmax(axis=-1)
                res.append(pred)
        mean_pred = np.mean(res)
        return mean_pred

class FF(nn):
    name = "Feed Forward Neural Network"
    
    def act(self, X, train):
        if len(X.shape) < 2: X = X[np.newaxis,:]
        output = [{'Z':None, 'X':X}]
        for l in self.layers:
            Z, X = l.act(X, train)
            output.append({'Z':Z,'X':X})
        self.cache = output

    def diff(self, y):
        for l in self.layers:
            l.ddbs = np.zeros(l.bs.shape)
            l.ddws = np.zeros(l.ws.shape)
        
        for n in range(-1,-len(self.layers)-1,-1):
            l = self.layers[n]
            if n == -1: 
                da = self.costf.diff(self.cache[n]['Z'], self.cache[n]['X'], y)
                l.diff(da, self.cache[n-1]['X'])
            else:
                da = l.diff(da, self.cache[n-1]['X'], self.layers[n+1], self.cache[n]['Z'])