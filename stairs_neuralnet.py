#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 18 16:53:17 2020

@author: vamsi
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class NeuralNetwork():
    def __init__(self,sizes):
        self.layers = sizes
        self.weights = [np.random.rand(x+1,y) for x,y in zip(sizes[:-1],sizes[1:])]
        self.delta_w = []
    def sigmoid(self,z):
        return 1/(1+np.exp(-z))
    def softmax(self,z):
        z = np.exp(z)        
        summ = np.sum(z,axis=1)
        summ = np.concatenate((np.vstack(summ),np.vstack(summ)),axis = 1)
        return z/summ
    def targets_y(self,data_y):
        max = np.max(data_y)
        self.Y = np.zeros((self.X.shape[0], max + 1))
        for row, data in enumerate(data_y):
            self.Y[row][data] = 1
    def train_data(self,data_x,data_y):
        cost = 0
        acc = 0
        self.X = np.array(data_x)
        self.targets_y(data_y)
        self.feedforward(self.X)
        for r,t in enumerate(self.Y):
            cost += -(t[0] * np.log(self.activations[-1][r][0]) + (1-t[1]) * np.log(1 - self.activations[-1][r][1]))
            if np.argmax(t) == np.argmax(self.activations[-1][r]):
                acc += 1
        self.backpropogation(1)
        self.delta_w.reverse()
        for row, nabla_w in enumerate(self.delta_w):
            self.weights[row] -= (0.1) * nabla_w 
        return acc,cost/15
    def feedforward(self,x):
        x = x/255
        self.activations = []
        self.delta_w = []
        for i,j in enumerate(self.layers[:-2]):
            n = x.shape[0]            
            x = np.concatenate((np.vstack([1]*n),x),axis = 1)
            self.activations.append(x)
            z = np.dot(x,self.weights[i])
            x = self.sigmoid(z)
        n = x.shape[0]            
        x = np.concatenate((np.vstack([1]*n),x),axis=1)
        self.activations.append(x)
        z = np.dot(x,self.weights[i+1])
        softmax = self.softmax(z)
        self.activations.append(softmax)
    def backpropogation(self,i):        
        error = self.activations[-1] - self.Y
        self.delta_w.append(np.dot(self.activations[-2].T,error))
        length = len(self.activations)
        for i in range(1,length-1):
            x_layer = self.activations[-1-i][:,1:] * (1 - self.activations[-1-i][:,1:])
            wts = np.dot(error,self.weights[-i].T)
            x_layer = wts[:,1:] * x_layer
            self.delta_w.append(np.dot(self.activations[-1-(i+1)].T,x_layer))
    def test_data(self,data_x,data_y):
        self.feedforward(data_x)
        self.X = np.array(data_x)
        targets = self.activations[-1]
        self.targets_y(data_y)
        rate = 0
        for i,target in enumerate(targets):
            if np.argmax(self.Y[i]) == np.argmax(target):
                rate+=1
        print("Accuracy Rate is ",(rate/self.X.shape[0]))
if __name__ == '__main__':
    df1 = pd.read_csv('train.csv')
    df2 = pd.read_csv('test.csv')
    net = NeuralNetwork([4,3,2])
    train_XY = df1.values
    test_no = train_XY.shape[0]
    testd = int(test_no/15)
    cost = []
    accuracy = []
    for i in range(200):
        np.random.shuffle(train_XY)
        j = 0
        costs = 0
        acc = 0
        for i in range(testd):
            tdata_XY = train_XY[j:j+15,:]
            np.random.shuffle(tdata_XY)
            train_X = tdata_XY[:,1:-1]
            train_Y = tdata_XY[:,-1]
            j = j + 15
            a,c= net.train_data(train_X,train_Y)
            costs+= c
            acc+=a
        cost.append(costs)
        accuracy.append((acc/train_XY.shape[0])*100)
    plt.title('Costs')
    plt.plot(cost)
    plt.show()
    plt.title('Accuracy')
    plt.plot(accuracy)
    plt.show()
    
    test_XY = df2.values
    test_X = test_XY[:,1:-1]
    test_Y = test_XY[:,-1]
    net.test_data(test_X,test_Y)
    
    
    
    
