import numpy as np

def import_MNIST(morph:str = None) -> tuple:
    """
    Using the sklearn to download the openml MNIST 784 dataset. 
    https://www.openml.org/d/554
    
    Returns
    -------
    tuple
        
        The tuple has three elements:
            train dataset with 50000 examples
            validation dataset with 10000 examples
            test dataset with 10000 examples
        
        Each dataset has an X and an y component. The 
                                                                                                                                                                                            
    """
    from sklearn.datasets import fetch_openml
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
    X = X/X.max()
    if morph == 'norm':
        X = (X-X.mean())/X.std()
    y = util.onehot(y)
    return (X[:50000], y[:50000]),(X[50000:60000], y[50000:60000]), (X[60000:], y[60000:])

def import_SIN():
    X = []
    y =[]
    t = np.arange(1150)
    a = np.around(np.sin(t),2)*100
    a_hot = util.onehot(a)
    for i in range(len(a_hot)-50):
        X.append(a_hot[i:i+50])
        y.append(a_hot[i+50])
    X = np.array(X)
    y = np.array(y)
    return (X[:1000], y[:1000]), (X[1000:1050], y[1000:1050]), (X[1050:1100], y[1050:1100])


def import_SEQS():
    np.random.seed(42)
    samples = []
    word2inx = util.Word2Inx()
    X, y = [], []
    
    for _ in range(1000): 
        num_tokens = np.random.randint(1, 10)
        sample = ['a'] * num_tokens + ['b'] * num_tokens + ['EOS']
        samples.append(sample)

    for sample in samples:
        sample = [word2inx(i) for i in sample]
        sample = util.onehot(sample, 4)
        X.append(sample[:-1])
        y.append(sample[1:])
    X = np.array(X)
    y = np.array(y)
    return (X[:800], y[:800]), (X[800:900], y[800:900]), (X[900:], y[900:])

class util:
    @staticmethod
    def accuracy(y_true, y_pred):
        return np.mean(y_pred.argmax(axis=-1) == y_true.argmax(axis=-1))
    @staticmethod
    def onehot(y, vocab_size=None):
        try: 
            y = np.array(y, int)
            if not vocab_size:
                vocab_size = np.max(y)+1
            res = np.zeros([y.size, vocab_size])
            res[range(y.size), y] = 1.
        except ValueError:
                raise NotImplementedError("onehot only works with int input")
        return res
    
    @staticmethod
    def int_sqrt(x): 
        if x==int(x**0.5)**2: 
            return int(x**0.5)
        else: 
            raise ValueError("Input not squre number")
    
    @staticmethod
    def mindim(X,dims=2):
        if len(X.shape) < dims: 
            X = X[np.newaxis,:]
        return X
            
    class Word2Inx:
        def __init__(self):
            self.vocab = {}
            self.index = 0
            
        def __call__(self, word):
            if word not in self.vocab.keys():
                self.vocab[word] = self.index
                self.index += 1
            return self.vocab[word]
    
