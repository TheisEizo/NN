#import os
#os.chdir('C:/Users/45414/Python/nn')

from nn_v1_func import import_MNIST, FF
from nn_v1_neurons import Sigmoid
from nn_v1_layers import FullCon, Dropout
from nn_v1_cost import CrossEntropy

#(X_train, y_train), (X_test, y_test) = import_MNIST()
#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)

NN.SGD(X_train,y_train)
#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,30)),
        FullCon(Sigmoid, (30,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
        CrossEntropy)

NN.SGD(X_train,y_train)
#%% WORKS
NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        Dropout(0.9),
        FullCon(Sigmoid, (30,10)),
        ], 
    CrossEntropy)
NN.SGD(X_train,y_train)
