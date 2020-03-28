import numpy as np

def import_MNIST():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X/X.max()
    X = (X-X.mean())/X.std()
    return (X[:60000], y[:60000]), (X[60000:], y[60000:])

class util:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_pred.argmax(axis=-1) == y_true.argmax(axis=-1))
    @staticmethod
    def onehot(y):
        y = np.array(y, int)
        res = np.zeros([y.size, np.max(y) + 1])
        res[range(y.size), y] = 1.
        return res
    
class nn:
    name = "Basic Neural Net"
    def __init__(self, layers, costf):
        self.layers = layers
        self.costf = costf()
        self.cache = None
        
    def __repr__(self):
        return f'{self.name} with {self.layers} and {self.costf}'
        
    def SGD(self, X, y, epochs=10, batch_size=10, eta=0.1, lmbda=0.0, test_data=None):
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
                self.update(batch, eta, lmbda, n)
            print(f'Epoch {i+1} out of {epochs}')
            self.act(X)
            print('Accuracy on training data: '+str((util.accuracy(y, self.cache[-1]['X']))))
            if test_data: 
                self.act(X_t)
                print('Accuracy on test data: '+str((util.accuracy(y_t, self.cache[-1]['X']))))
                
    def update(self, batch, eta, lmbda, n):
        for l in self.layers:
            l.dbs = np.zeros(l.bs.shape); l.dws = np.zeros(l.ws.shape)
        
        for X, y in zip(batch[0], batch[1]):
            self.act(X)
            self.diff(y)
            for l in self.layers:
                l.dbs += l.ddbs; l.dws += l.ddws
        for l in self.layers:
            l.ws -= eta*(lmbda/n)*l.ws
            l.ws -= eta/len(batch)*l.dbs 
            l.ws -= eta/len(batch)*l.dws
            
    def predict(self, X):
        self.act(X)
        return self.cache[-1]['X'].argmax(axis=-1)
        
    def act(X): raise NotImplementedError
    def diff(self, y): raise NotImplementedError  
        
class FF(nn):
    name = "Feed Forward Network"
    def act(self, X):
        if len(X.shape) < 2: X = X[np.newaxis,:]
        output = [{'Z':None, 'X':X}]
        for l in self.layers:
            Z, X = l.act(X)
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