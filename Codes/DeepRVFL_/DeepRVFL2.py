# -*- coding: utf-8 -*-
"""
Created on Thu Jul 22 00:07:31 2021

@author: ruobi
"""

import numpy as np
import random
# from sklearn.svm import LinearSVR
# from sklearn.linear_model import Ridge,Lasso,ElasticNet
def sigmoid(x):

    z = 1/(1 + np.exp(-x)) 
    return z
class DeepRVFL2(object):
    def __init__(self, Nu,Nh,Nl, configs, verbose=0):
        self.W = {} # recurrent weights
        self.Win = {} # recurrent weights
        #self.Gain = {} # activation function gain
        self.Bias = {} # activation function bias

        self.Nu = Nu # number of inputs
        self.Nh = Nh # number of units per layer
        self.Nl = Nl
        iss=configs.iss
        # sparse recurrent weights init
        self.readout = configs.readout
        if configs.enhConf.connectivity < 1:
            for layer in range(Nl):
                self.W[layer] = np.zeros((Nh,Nh))
                for row in range(Nh):
                    number_row_elements = round(configs.enhConf.connectivity * Nh)
                    row_elements = random.sample(range(Nh), number_row_elements)
                    self.W[layer][row,row_elements] = np.random.uniform(-1,+1, size = (1,number_row_elements))
                    
        # full-connected  weights init      
        else:
            for layer in range(Nl):
                self.W[layer] = np.random.uniform(-1,+1, size = (Nh,Nh))
        for layer in range(Nl):
            input_scale = iss[layer]

            if layer==0:
                self.Win[layer] = np.random.uniform(-input_scale, input_scale, size=(Nh,Nu+1))
            else:
                self.Win[layer] = np.random.uniform(-input_scale, input_scale, size=(Nh,Nh+Nu+1))
                
            self.Bias[layer] = np.zeros((Nh,1))
            
        self.input_scale = iss[layer]
    
    def computeLayerState(self,  layer,x_raw,inistate=None, DeepIP = 0):
        #Win Nh*Nu
        #x_raw Nu*Nsampl
        #inistate= Nh*1
        Ns=x_raw.shape[1]
        state = np.zeros((self.Nh,Ns))
        
        if layer==0:
            self.Win[layer] = self.Win[layer][:,:x_raw.shape[0]+1]
            state=sigmoid(self.Win[layer][:,:-1].dot(x_raw))+np.tile(np.expand_dims(self.Win[layer][:,-1],1),[1,Ns])
        else:
            # print( x_raw.shape,inistate.shape)
            x_cat=np.concatenate((x_raw,inistate),axis=0)
            self.Win[layer] = self.Win[layer][:,:x_cat.shape[0]+1]#np.random.uniform(-self.input_scale, self.input_scale, size=(self.Nh,x_cat.shape[0]+1))
            state=sigmoid(self.Win[layer][:,:-1].dot(x_cat))+np.tile(np.expand_dims(self.Win[layer][:,-1],1),[1,Ns])
        return state

    def computeGlobalState(self,x_raw):
        Ns=x_raw.shape[1]
        state=np.zeros((self.Nl*self.Nh,Ns))
        for layer in range(self.Nl):
            if layer==0:
                state[:self.Nh,:]=self.computeLayerState(layer,x_raw)
            else:
                s=state[self.Nh*layer:self.Nh*(layer+1),:]
                state[self.Nh*layer:self.Nh*(layer+1),:]=self.computeLayerState(layer,x_raw,s)
        return state
    def trainReadout(self,trainStates,trainTargets,lb, verbose=0,SVR_para=None,l1_ratio=0.1):
        #trainStates (nh,nsample)
        #traintargets (1,nsample)
        Ns=trainStates.shape[1]
        X=np.ones((trainStates.shape[0]+1,Ns))
        X[:-1,:] = trainStates    
        trainStates = X
        if self.readout.trainMethod == 'SVD': # SVD, accurate method
#            print('esn',trainStates.shape,trainTargets.shape)
            U, s, V = np.linalg.svd(trainStates, full_matrices=False);  
#            print(s.shape)
            s = s/(s**2 + lb)
        else:
            # print(trainStates.shape,trainTargets.shape)
            # B = trainTargets.dot(trainStates.T)
            # A = trainStates.dot(trainStates.T)

            # self.Wout = np.linalg.solve((A + np.eye(A.shape[0], A.shape[1]) * lb), B.T).T
            x=trainStates
            xTx=np.dot(x,x.T)
            ridge_xTx=xTx+lb * np.eye(x.shape[0])
            ridge_inv=np.linalg.pinv(ridge_xTx)
            self.Wout=np.dot(np.dot(ridge_inv,x),trainTargets.T).T
            # self.Wout=np.dot(ridge_inv,np.dot(x,trainTargets.T)).T
            # print(self.Wout.shape)
            # return se
    def computeOutput(self,state):
        # print(state.shape,self.Wout.shape)
        return self.Wout[:,0:-1].dot(state) + np.expand_dims(self.Wout[:,-1],1)