""" GAN TEST """
import matplotlib.pyplot as plt
import numpy as np
#%%
digit = 0
train_part = (train[0][np.where(train[1][:,digit]==1)[0]],
              train[1][np.where(train[1][:,digit]==1)[0]])
#%% Works
Gen = FF(
        [FullCon(ReLU, (100,128)),
        FullCon(Tanh, (128, 28*28))], 
        GANCrossEntropy)
Dis = FF(
        [FullCon(LReLU, (28*28,128)),
        FullCon(Sigmoid, (128,1))], 
        GANCrossEntropy)
NN = GAN(Gen, Dis, GANCrossEntropy)
NN.SGD(train_part, epochs=3, eta=1e-3, printoutImage=True)
#%% Works
Gen = FF(
        [FullCon(Sigmoid, (100,128)),
        FullCon(Sigmoid, (128, 28*28))], 
        GANCrossEntropy)
Dis = FF(
        [FullCon(Sigmoid, (28*28,128)),
        FullCon(Sigmoid, (128,1))], 
        GANCrossEntropy)
NN = GAN(Gen, Dis, GANCrossEntropy)
NN.SGD(train_part, epochs=3, eta=1e-3, printoutImage=True)
#%%
Gen = FF(
        [FullCon(Sigmoid, (100,128)),
        FullCon(Sigmoid, (128, 28*28))], 
        GANCrossEntropy)
Dis = FF(
        [FullCon(Sigmoid, (28*28,128)),
        FullCon(Sigmoid, (128,1))], 
        GANCrossEntropy)
NN = GAN(Gen, Dis, GANCrossEntropy)
NN.SGD(train_part, epochs=2, eta=1e-3, momentum=0.1, printoutImage=True)
#%%
NN.SGD(train_part, epochs=10, eta=1e-1, printoutImage=True)
#%%
NN.SGD(train_part, epochs=10, eta=1e-2, printoutImage=True)
#%%
NN.SGD(train_part, epochs=10, eta=1e-3, printoutImage=True)
#%%
NN.SGD(train_part, epochs=10, eta=1e-4, printoutImage=True)
#%%
NN.SGD(train_part, epochs=10, eta=1e-5, printoutImage=True)
#%%
NN.SGD(train_part, epochs=10, eta=1e-6, printoutImage=True)
#%%
NN.SGD(train_part, epochs=10, eta=1e-7, printoutImage=True)
#%%
z = NN.generate(1)
plt.imshow(z.reshape(28,28))
#%%
z = train_part[0][np.random.randint(0, len(train_part))]
plt.imshow(z.reshape(28,28))

#%%
from sklearn.datasets import fetch_openml
#X, y = fetch_openml('mnist_784', version=1, return_X_y=True)
X_train, y_train = X[:60000].reshape((-1,28,28)), y[:60000].astype(int)
#%%
from model import GAN
numbers = [i for i in range(10)]
model = GAN(numbers, learning_rate=1e-3, decay_rate=1e-4, epochs=100)
J_Ds, J_Gs = model.train(X_train, y_train)
