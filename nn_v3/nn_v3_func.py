import numpy as np

def import_MNIST():
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X/X.max()
    X = (X-X.mean())/X.std()
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

    
