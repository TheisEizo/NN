import os
os.chdir('C:/Users/45414/Python/nn/nn_v2')

from nn_v2_func import import_MNIST
##Expand dataset
from nn_v2_neurons import Sigmoid, Tanh, ReLU, Softmax, Identity
from nn_v2_layers import FullCon, Dropout, Conv, Pool
from nn_v2_cost import CrossEntropy
from nn_v2_networks import FF
from nn_v2_reg import L1, L2

#train, validate, test = import_MNIST()
#%%
NN = FF([
        Conv(Identity, (25,3)),
        Pool('max',(3,-1),(2,2)),
        FullCon(Sigmoid, (3*13*13,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1])


#%% DOESN'T WORK
NN = FF([
        FullCon(ReLU, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1])
#%% DOESN'T WORK
NN = FF([
        FullCon(Softmax, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1])

#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1], test_data = test)