import os
os.chdir('C:/Users/45414/Python/nn')

from nn_v1_func import *
(X_train, y_train), (X_test, y_test) = import_MNIST()

NN = FF([
        FullCon(Sigmoid, (28*28,30)),
        FullCon(Sigmoid, (30,10)),
        ], 
    CrossEntropy)

NN.SGD(X_train,y_train,test_data=(X_test,y_test))