import os
os.chdir('C:/Users/45414/Python/nn/nn_v3')

from nn_v3_func import import_MNIST
train, validate, test = import_MNIST()
#%%
from nn_v3_func import import_SIN
train_sin, validate_sin, test_sin = import_SIN()
#%%
from nn_v3_neurons import Sigmoid, Tanh, ReLU, LReLU, Softmax, Identity
from nn_v3_layers import FullCon, Dropout, Conv, Pool, Networks, RecurrentFullCon
from nn_v3_cost import CrossEntropy, SquaredLoss
from nn_v3_networks import FF
from nn_v3_reg import L1, L2
#%% DOESNT LEARN

NN = FF(
        [RecurrentFullCon(Tanh, Softmax, (101, 10, 101), 5)],
        CrossEntropy)
NN.SGD(train_sin,test_sin, epochs=100, printout='Loss')
#%% WORKS
NNS = [
       FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy) 
        for i in range(5)]

NN = FF(
        [Networks(Sigmoid, NNS)],
        CrossEntropy)

NN.SGD(train)
#%% WORKS
NN = FF(
        [Conv(Identity, (25,3)),
        Pool('mean',(3,-1),(2,2)),
        Dropout('binomial',0.9),
        FullCon(Sigmoid, (3*13*13,10))], 
        CrossEntropy)
NN.SGD(train)
#%% WORKS
NN = FF(
        [FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10))], 
        CrossEntropy)
NN.SGD(train, printout='Loss')
#%% TO DO
#Fix this
    ##%% DOESN'T LEARN???
    #NN = FF(
    #        [FullCon(ReLU, (28*28,30)),
    #        FullCon(Sigmoid, (30,10))], 
    #        CrossEntropy)
    #NN.SGD(train)
    ##%% LEARN
    #NN = FF(
    #        [FullCon(LReLU, (28*28,30)),
    #        FullCon(Sigmoid, (30,10))], 
    #        CrossEntropy)
    #NN.SGD(train)
    
#Expand dataset

# Fix BatchNorm