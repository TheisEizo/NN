# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 09:31:20 2020

@author: Theis Eizo
"""
#%%
import numpy as np
a = np.arange(2*16).reshape(2,16)

def int_sqrt(x): 
    if x==int(x**0.5)**2: return int(x**0.5)
    else: raise ValueError("Input not squre number")

def im2col(X, padding, kshape, roll):
    p_shape = ((0,0),)*(len(X.shape)-2)+((padding[0],)*2,)*2
    if padding[1] == 'min': 
        X = np.pad(X, p_shape, 'constant', constant_values=X.min())
    else:
        X = np.pad(X, p_shape, 'constant', constant_values=padding[1])
    
    ## Filtering
    
    X = np.array([np.roll(X, -i*roll, axis=-1)[:,:,:kshape[-1]] 
                for i in range(0,X.shape[-1]-kshape[-1]+1,roll)])
    X = np.array([np.roll(X, -i*roll, axis=-2)[:,:,:kshape[-2]] 
                for i in range(0,X.shape[-2]-kshape[-2]+1,roll)])
    
    X = X.reshape(X.shape[:-2]+(-1,))
    X = X.reshape((-1,)+X.shape[-2:])
    X = np.moveaxis(X,-2,0)
    return X

def conv_act(X):
    ws = np.random.randn(4*4, 3)/np.sqrt(4*4)
    bs = np.random.randn(1, 3)

    roll = 1
    imshape = None
    kshape = None
    padding=(1, 0) #Padding with, paddig value (float or 'min')

    if imshape == None: imshape = (int_sqrt(X.shape[-1]),)*2
    if kshape == None: kshape = (int_sqrt(ws.shape[0]),)*2
    
    X = X.reshape(X.shape[:-1]+imshape)
    ## Padding according to padding
    X = im2col(X, padding, kshape, roll)
    
    #X = np.dot(X, ws)+bs
    #X = X.reshape(X.shape[0],-1)
    return X

a = conv_act(a)

#%%
def im_split(X, kshape):
    X = np.array(np.split(X,X.shape[-1]/kshape[-1],axis=-1))
    X = np.array(np.split(X,X.shape[-2]/kshape[-2],axis=-2))
    X = np.moveaxis(X,2,0)
    X = X.reshape((X.shape[0],)+(-1,)+X.shape[3:])
    X = np.moveaxis(X,1,2)
    return X

def pool_act(X):

    ptype = 'mean'
    imshape = (4,4)
    kshape = (2,2)
    if imshape == None: imshape = (int_sqrt(X.shape[-1]),)*2
    elif imshape[-1]==-1: imshape = (imshape[0],)+(int_sqrt(X.shape[-1]/imshape[0]),)*2
    X = X.reshape(X.shape[:-1]+imshape)
    X = im_split(X, kshape)
    if ptype == 'max':
        X = np.amax(X, axis=(-2,-1))
    if ptype == 'mean':
        X = np.mean(X,axis=(-2,-1))
    return X

pool_act(a)
#%%
import cv2
