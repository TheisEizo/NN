import os
os.chdir('C:/Users/45414/Python/nn/nn_v3')

from nn_v3_func import import_MNIST #,util
train, validate, test = import_MNIST()
##TO DO: Expand dataset
#%%
from nn_v3_neurons import Sigmoid, Tanh, ReLU, LReLU, Softmax, Identity
from nn_v3_layers import FullCon, Dropout, Conv, Pool, Networks
from nn_v3_cost import CrossEntropy, SquaredLoss
from nn_v3_networks import FF
from nn_v3_reg import L1, L2
#%% WORKS
NN = FF(
        [Conv(Identity, (25,3)),
        Pool('mean',(3,-1),(2,2)),
        Dropout('binomial',0.9),
        FullCon(Sigmoid, (3*13*13,10))], 
        CrossEntropy)
NN.SGD(train, test)
#%% 
NN1 = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN2 = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN3 = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN4 = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN5 = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NNS = FF(
        [Networks(Sigmoid, [NN1, NN2, NN3, NN4, NN5]),],
        CrossEntropy)
NNS.SGD(train, test, epochs=10, reg=L2(5))
#%% WORKS
NN = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN.SGD(train, test)
#%% DOESN'T LEARN
NN = FF(
        [FullCon(ReLU, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN.SGD(train)
