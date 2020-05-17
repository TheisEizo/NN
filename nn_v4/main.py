import numpy as np
import os
#os.chdir('C:/Users/45414/Python/nn/nn_v4')
os.chdir('/home/theizo/Python/nn/nn_v4')
#%% Load MNIST dataset 
from func import import_MNIST
train, validate, test = import_MNIST()
#%%
part_train = (train[0][:1000], train[1][:1000])
#%% Load Sine Wave dataset
from func import import_SIN
train, validate, test = import_SIN()
#%% Load Char Seqs dataset
from func import import_SEQS
train, validate, test =import_SEQS()
#%% Load all nn parts
from neurons import Sigmoid, Tanh, ReLU, LReLU, Softmax, Identity
from layers import FullCon, Dropout, Conv, Pool, Networks
from layers import RecurrentFullCon, LSTMFullCon, GRUFullCon
from cost import CrossEntropy, SquaredLoss, RNNCrossEntropy, GANCrossEntropy
from networks import FF, GAN
from reg import L1, L2
#%% GAN: WORKS
digits = [0]
train_part = (train[0][np.where(train[1][:,digits]==1)[0]],
              train[1][np.where(train[1][:,digits]==1)[0]])
#%%
Gen = FF(
        [FullCon(ReLU, (100,128)),
        FullCon(Tanh, (128, 28*28))], 
        GANCrossEntropy)
Dis = FF(
        [FullCon(LReLU, (28*28,128)),
        FullCon(Sigmoid, (128,1))], 
        GANCrossEntropy)
NN = GAN(Gen, Dis, GANCrossEntropy)
NN.SGD(train_part, epochs=100, eta=1e-3, eta_decay=5e-2, printoutImage=True)
#%% GRU: WORKS
NN = FF(
        [GRUFullCon(Tanh, Sigmoid, Softmax, (4, 50, 4))],
        RNNCrossEntropy)
NN.SGD(train, eta=1e-3)
#%% LSTM: WORKS
NN = FF(
        [LSTMFullCon(Tanh, Sigmoid, Softmax, (4, 50, 4))],
        RNNCrossEntropy)
NN.SGD(train, eta=1e-3)
#%% RNN: WORKS
NN = FF(
        [RecurrentFullCon(Tanh, Softmax, (4, 50, 4))],
        RNNCrossEntropy)
NN.SGD(train, eta=1e-4)
#%% Networks: WORKS
NNS = [
       FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy) 
        for i in range(5)]
NN = FF(
        [Networks(Sigmoid, NNS)],
        CrossEntropy)
NN.SGD(part_train)
#%% Conv+Pool+Dropout: WORKS
NN = FF(
        [Conv(Identity, (25,3)),
        Pool('mean',(3,-1),(2,2)),
        Dropout('binomial',0.9),
        FullCon(Sigmoid, (3*13*13,10))], 
        CrossEntropy)
NN.SGD(part_train)
#%% FullCon: WORKS
NN = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN.SGD(part_train)
#%% TO DO
# Expand dataset

# Apply BatchNorm Layer

# Make sure that rnns also work in a multiple layers setting 
# (check backprop da)

# Make GAN network

# Make documentation
#%%