import numpy as np

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
            printout="Accuracy",
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
                
    def update(self, batch, eta, n, reg, momentum):
        for l in self.layers:
            if momentum and 'vs' not in dir(l): 
                l.vs = np.zeros(l.ws.shape)
            l.init_dws_dbs()
        for X, y in zip(batch[0], batch[1]):
            self.act(X, train=True)
            self.diff(y)
            for l in self.layers:
                l.update_dws_dbs()
        for l in self.layers:
            l.update_ws_bs(batch, eta, n, reg, momentum)
            l.clean()
        del self.cache
    
    def loss(self, X, y):
        self.act(X, train=False)
        loss = self.costf.act(self.cache[-1]['Z'], self.cache[-1]['X'], y)
        del self.cache
        return loss
    
        return
    def predict(self, X):
        self.act(X, train=False)
        res = self.cache[-1]['X'].argmax(axis=-1)
        del self.cache
        return res
    
    def accuracy(self, X, y):
        #This if loop is just a memory fix
        if (len(y) <= 1000) or (len(y)/1000 != len(y)//1000):
            y_pred = self.predict(X)
            res = (y_pred == y.argmax(axis=-1))
        else: 
            res = []           
            for Xi, yi in zip(np.split(X, 50), np.split(y, 50),):
                yi_pred = self.predict(Xi)
                pred = (yi_pred == yi.argmax(axis=-1))
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
        if len(X.shape) < 2: 
            X = X[np.newaxis,:]
        output = [{'Z':None, 'X':X}]
        for l in self.layers:
            Z, X = l.act(X, train)
            output.append({'Z':Z,'X':X})
        self.cache = output

    def diff(self, y):        
        for n in range(-1,-len(self.layers)-1,-1):
            l = self.layers[n]
            if n == -1: 
                da = self.costf.diff(self.cache[n]['Z'], 
                                     self.cache[n]['X'], 
                                     y)
                da = l.diff(da, self.cache[n-1]['X'])
            else:
                da = l.diff(da, 
                            self.cache[n-1]['X'], 
                            self.layers[n+1], 
                            self.cache[n]['Z'])