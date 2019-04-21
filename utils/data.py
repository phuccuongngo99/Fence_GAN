#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 20 16:40:28 2019

@author: deeplearning
"""
import numpy as np
from keras.datasets import mnist

def preprocess(x):
    x = np.float64(x)
    x = (x / 255 - 0.5) * 2
    x = np.clip(x, -1, 1)
    return x

def load_data(args):
    ###Dataset: 2d, mnist, cifar10, kdd99, custom
    ###Ano_class: 1 actual class label of the dataset
    if args.dataset == 'mnist':
        return get_mnist(args.ano_class)
    #elif dataset == 'cifar10':
        

def get_mnist(ano_digit):
    (x_tr, y_tr), (x_tst, y_tst) = mnist.load_data()
    
    x_total = np.concatenate([x_tr, x_tst])
    y_total = np.concatenate([y_tr, y_tst])
    print(x_total.shape, y_total.shape)
    
    x_total = x_total.reshape(-1, 28, 28, 1)
    x_total = preprocess(x_total)
    
    ###80% of real data used in training, 
    ###20% of real data, all anomalous data used for validation and testing
    ###Validation is 30%
    ###Create anomalous data for testing
    delete = []
    for count, i in enumerate(y_total):
        if i == ano_digit:
            delete.append(count)
    
    ano_data = x_total[delete,:]
    real_data = np.delete(x_total,delete,axis=0)
    del x_total
    
    x_train = real_data[:int(0.8*real_data.shape[0]),...]
    
    x_test = np.concatenate((real_data[int(0.8*real_data.shape[0]):int(0.95*real_data.shape[0]),...], ano_data[:ano_data.shape[0]*3//4]))
    y_test = np.concatenate((np.ones(int(0.95*real_data.shape[0])-int(0.8*real_data.shape[0])), np.zeros(ano_data.shape[0]*3//4)))
    
    x_val = np.concatenate((real_data[int(0.95*real_data.shape[0]):,...], ano_data[ano_data.shape[0]*3//4:]))
    y_val = np.concatenate((np.ones(real_data.shape[0]-int(0.95*real_data.shape[0])), np.zeros(ano_data.shape[0]-ano_data.shape[0]*3//4)))
    
    return x_train, x_test, y_test, x_val, y_val, ano_data

#def get_cifar10():

#def get_kdd99():
    
#def get_custom():