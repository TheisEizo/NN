import os
os.chdir('C:/Users/45414/Python/nn/nn_v1')

from nn_v1_func import import_MNIST
from nn_v1_neurons import Sigmoid, Tanh, ReLU
from nn_v1_layers import FullCon, Dropout
from nn_v1_cost import CrossEntropy
from nn_v1_networks import FF
from nn_v1_reg import L1, L2
#train, validate, test = import_MNIST()



#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1], momentum=0.5)
#%% DOESN'T WORK
NN = FF([
        FullCon(ReLU, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1])
#%% WORKS
NN = FF([
        FullCon(Tanh, (28*28,30)),
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
NN.SGD(train[0], train[1], reg=L2(0.5))
#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1], reg=L1(0.5))
#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        Dropout(0.95),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)
NN.SGD(train[0], train[1])

#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,30)),
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