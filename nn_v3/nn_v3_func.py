import numpy as np

def import_MNIST():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X/X.max()
    X = (X-X.mean())/X.std()
    y = util.onehot(y)
    return (X[:50000], y[:50000]),(X[50000:60000], y[50000:60000]), (X[60000:], y[60000:])

def import_SIN():
    X = []
    y =[]
    t = np.linspace(0,np.pi*2,100)
    freqs = np.linspace(-5,-5,740)
    
    for freq in freqs:
        a = np.around(np.sin(freq*t)/2+1/2,2)*100
        a_hot = util.onehot(a)
        for i in range(len(a_hot)-5):
            X.append(a_hot[i:i+5])
            y.append(a_hot[i+5])
    X = np.stack(X)[:70000].reshape((70000,-1))
    y = np.stack(y)[:70000]
    return (X[:50000], y[:50000]),(X[50000:60000], y[50000:60000]), (X[60000:], y[60000:])

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
    
    @staticmethod
    def int_sqrt(x): 
        if x==int(x**0.5)**2: 
            return int(x**0.5)
        else: 
            raise ValueError("Input not squre number")

    
