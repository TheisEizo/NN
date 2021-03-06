import os
#os.chdir('C:/Users/45414/Python/nn/nn_v3')
os.chdir('/home/theizo/Python/nn/nn_v3')
#%% Load MNIST dataset 
from func import import_MNIST
train, validate, test = import_MNIST()
part_train = (train[0][:100], train[1][:100])
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
from cost import CrossEntropy, SquaredLoss, RNNCrossEntropy
from networks import FF
from reg import L1, L2
#%% GRU: 
NN = FF(
        [GRUFullCon(Tanh, Sigmoid, Softmax, (4, 50, 4))],
        RNNCrossEntropy)
NN.SGD(train, epochs=1, eta=1e-3)
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
#Expand dataset

# Apply BatchNorm