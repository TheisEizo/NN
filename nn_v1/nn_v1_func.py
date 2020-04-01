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
    
    @staticmethod
    def im_split(X, kshape):
        X = np.array(np.split(X,X.shape[-1]/kshape[-1],axis=-1))
        X = np.array(np.split(X,X.shape[-2]/kshape[-2],axis=-2))
        X = np.moveaxis(X,2,0)
        X = X.reshape((X.shape[0],)+(-1,)+X.shape[3:])
        X = np.moveaxis(X,1,2)
        return X