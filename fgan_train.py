import numpy as np
from numpy.random import seed

from tensorflow import set_random_seed

from tqdm import trange

import keras.backend as K

import matplotlib.pyplot as plt
from collections import OrderedDict
import os
import json
import random

from utils.model import *
from utils.data import load_data
from utils.visualize import show_images, compute_au, histogram

from sklearn.metrics import roc_auc_score


def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def noise_data(n_samples, latent_dim):
        return np.random.normal(0,1,[n_samples,latent_dim])

    
def D_data(n_samples,G,mode,x_train,latent_dim):
    #Feeding training data for normal case
    if mode == 'normal':
        sample_list = random.sample(list(range(np.shape(x_train)[0])), n_samples)
        x_normal = x_train[sample_list,...]
        y1 = np.ones(n_samples)
        
        return x_normal, y1
    
    #Feeding training data for generated case    
    if mode == 'gen':
        noise = noise_data(n_samples, latent_dim)
        x_gen = G.predict(noise)
        y0 = np.zeros(n_samples)
        
        return x_gen, y0
    
def pretrain(args, G ,D, GAN, x_train, x_test, y_test, x_val, y_val):
    ###Pretrain discriminator
    ###Generator is not trained
    print("===== Start of Pretraining =====")
    batch_size = args.batch_size
    pretrain_epoch = args.pretrain
    latent_dim = args.latent_dim
    for e in range(pretrain_epoch):
        with trange(x_train.shape[0]//batch_size, ascii=True, desc='Pretrain_Epoch {}'.format(e+1)) as t:
            for step in t:                
                loss = 0
                set_trainability(D, True)
                K.set_value(gamma, [1])
                x,y = D_data(batch_size,G,'normal',x_train, latent_dim)
                loss += D.train_on_batch(x, y)
                
                set_trainability(D, True)
                K.set_value(gamma, [args.gamma])
                x,y = D_data(batch_size,G,'gen',x_train, latent_dim)
                loss += D.train_on_batch(x,y)
                
                t.set_postfix(D_loss=loss/2)
        print("\tDisc. Loss: {:.3f}".format(loss/2))
    print("===== End of Pretraining =====")
        
        
def train(args, G ,D, GAN, x_train, x_test, y_test, x_val, y_val):
    ###Adversarial Training
    epochs = args.epochs
    batch_size = args.batch_size
    v_freq= args.v_freq
    ano_class = args.ano_class
    dataset = args.dataset
    latent_dim = args.latent_dim

    if not os.path.exists('./result/{}/'.format(args.dataset)):
        os.makedirs('./result/{}/'.format(args.dataset))
    result_path = './result/{}/{}'.format(args.dataset,len(os.listdir('./result/{}/'.format(args.dataset))))
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    if dataset == 'mnist':
        d_loss = []
        g_loss = []
        best_val_prc = 0
        best_test_prc = 0
        
        for epoch in range(epochs):
            try:
                with trange(x_train.shape[0]//batch_size, ascii=True, desc='Epoch {}'.format(epoch+1)) as t:
                    for step in t:
                        ###Train Discriminator
                        loss_temp = []
                        
                        set_trainability(D, True)
                        K.set_value(gamma, [1])
                        x,y = D_data(batch_size,G,'normal',x_train, latent_dim)
                        loss_temp.append(D.train_on_batch(x, y))
                        
                        set_trainability(D, True)
                        K.set_value(gamma, [args.gamma])
                        x,y = D_data(batch_size,G,'gen',x_train, latent_dim)
                        loss_temp.append(D.train_on_batch(x,y))
                        
                        d_loss.append(sum(loss_temp)/len(loss_temp))
                        
                        ###Train Generator
                        set_trainability(D, False)
                        x = noise_data(batch_size, latent_dim)
                        y = np.zeros(batch_size)
                        y[:] = args.alpha
                        g_loss.append(GAN.train_on_batch(x,y))
                        
                        t.set_postfix(G_loss=g_loss[-1], D_loss=d_loss[-1])
            except KeyboardInterrupt: #hit control-C to exit and save video there
                break    
            
            if (epoch + 1) % v_freq == 0:
                val_prc, test_prc = compute_au(D, G, GAN, x_val, y_val, x_test, y_test, args.evaluation)
                
                f = open('{}/logs.txt'.format(result_path),'a+')
                f.write('\nEpoch: {}\n\tval_prc: {:.3f} \n\ttest_prc: {:.3f}'.format(epoch+1, val_prc, test_prc))
                f.close()
                
                if val_prc > best_val_prc:
                    best_prc = val_prc
                    best_test_prc = test_prc
                    histogram(G, D, GAN, x_test, y_test, result_path, latent_dim)
                    show_images(G.predict(noise_data(16, latent_dim)),result_path)
                    
                    G.save('{}/gen_anoclass_{}.h5'.format(result_path,ano_class))
                    D.save('{}/dis_anoclass_{}.h5'.format(result_path,ano_class))
                    
                print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}\n\tArea_prc: {:.3f}".format(g_loss[-1], d_loss[-1], val_prc))
            else:
                print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}".format(g_loss[-1], d_loss[-1]))
        
        print('==== End of Training ====')
        print('Dataset: {}| Anomalous class: {}| Best test {}: {}'.format(args.dataset, ano_class, args.evaluation, round(best_test_prc,3)))
        
        #Saving result in result.json file    
        result =[("best_test_prc",round(best_test_prc,3)),("val_prc",round(best_val_prc,3))]
        result_dict = OrderedDict(result)
        with open('{}/result.json'.format(result_path),'w+') as outfile:
            json.dump(result_dict, outfile, indent=4)

    elif dataset == 'cifar10':
        d_loss = []
        g_loss = []
        best_val_roc = 0
        best_test_roc = 0
        
        for epoch in range(epochs):
            try:
                with trange(x_train.shape[0]//batch_size, ascii=True, desc='Epoch {}'.format(epoch+1)) as t:
                    for step in t:
                        ###Train Discriminator
                        loss_temp = []
                        
                        set_trainability(D, True)
                        K.set_value(gamma, [1])
                        x,y = D_data(batch_size,G,'normal',x_train, dataset = dataset)
                        loss_temp.append(D.train_on_batch(x, y))
                        
                        set_trainability(D, True)
                        K.set_value(gamma, [args.gamma])
                        x,y = D_data(batch_size,G,'gen',x_train, dataset = dataset)
                        loss_temp.append(D.train_on_batch(x,y))
                        
                        d_loss.append(sum(loss_temp)/len(loss_temp))
                        
                        ###Train Generator
                        set_trainability(D, False)
                        x = noise_data(batch_size, dataset = 'cifar10')
                        y = np.zeros(batch_size)
                        y[:] = args.alpha
                        g_loss.append(GAN.train_on_batch(x,y))
                        
                        t.set_postfix(G_loss=g_loss[-1], D_loss=d_loss[-1])
                        
            except KeyboardInterrupt: #hit control-C to exit and save video there
                break
            
        if (epoch + 1 % v_freq == 0):
            
            pred_test = D.predict(x_test)
            pred_test = np.reshape(pred_test, x_test.shape[0])
            pred_val = D.predict(x_val)
            pred_val = np.reshape(pred_val, x_val.shape[0])
            
            val_roc = roc_auc_score(y_val, pred_val)
            test_roc = roc_auc_score(y_test, pred_test)
            
            if (val_roc > best_val_roc):
                best_val_roc = val_roc
                best_test_roc = test_roc
                
                histogram(G, D, GAN, x_test, y_test, result_path, latent_dim)
                generated_images = G.predict(noise_data(50, dataset = 'cifar10'))
                show_images(generated_images,epoch,result_path, dataset = 'cifar10')
                
                G.save('{}/gen_anoclass_{}.h5'.format(result_path,ano_class))
                D.save('{}/dis_anoclass_{}.h5'.format(result_path,ano_class))
                
            print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}\n\tArea_roc: {:.3f}".format(g_loss[-1], d_loss[-1], val_roc))
        else:
            print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}".format(g_loss[-1], d_loss[-1]))
                
        #Saving result in result.json file    
        result =[("best_test_roc",round(best_test_roc,3)),("val_roc",round(best_val_roc,3))]
        result_dict = OrderedDict(result)
        with open('{}/result.json'.format(result_path),'w+') as outfile:
            json.dump(result_dict, outfile, indent=4)

def training_pipeline(args):
    seed(args.seed)
    set_random_seed(args.seed)
    x_train, x_test, y_test, x_val, y_val = load_data(args)
    
    G, D, GAN = load_model(args)
    pretrain(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val)
    train(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val)