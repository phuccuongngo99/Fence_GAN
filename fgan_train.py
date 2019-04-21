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

from utils.model import load_model
from utils.data import load_data
from utils.visualize import show_images, D_test

from sklearn.metrics import roc_auc_score

gw = K.variable([1])

def set_trainability(model, trainable=False): #alternate to freeze D network while training only G in (G+D) combination
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable

def noise_data(n, dataset = 'mnist'):
    if dataset == 'mnist':
        return np.random.normal(0,1,[n,200])
    elif dataset == 'cifar10':
        return np.random.normal(0,1,[n, 256])

def D_data(n_samples,G,mode,x_train, dataset = 'mnist'):
    if mode == 'real':
        sample_list = random.sample(list(range(np.shape(x_train)[0])), n_samples)
        x_real = x_train[sample_list,...]
        y1 = np.ones(n_samples)
        
        return x_real, y1
        
    if mode == 'gen':
        if dataset == 'mnist':
            noise = noise_data(n_samples)
        elif dataset == 'cifar10':
            noise = noise_data(n_samples, dataset = 'cifar10')
        x_gen = G.predict(noise)
        y0 = np.zeros(n_samples)
        
        return x_gen, y0
    
def pretrain(args, G ,D, GAN, x_train, x_test, y_test, x_val, y_val, ano_data, dataset = 'mnist'):
    ###Pretrain discriminator
    ###Generator is not trained
    print("===== Start of Pretraining =====")
    batch_size = args.batch_size
    pretrain_epoch = args.pretrain
    for e in range(pretrain_epoch):
        with trange(x_train.shape[0]//batch_size, ascii=True, desc='Pretrain_Epoch {}'.format(e+1)) as t:
            for step in t:                
                loss = 0
                set_trainability(D, True)
                K.set_value(gw, [1])
                x,y = D_data(batch_size,G,'real',x_train, dataset = dataset)
                loss += D.train_on_batch(x, y)
                
                set_trainability(D, True)
                K.set_value(gw, [args.gamma])
                x,y = D_data(batch_size,G,'gen',x_train, dataset = dataset)
                loss += D.train_on_batch(x,y)
                
                t.set_postfix(D_loss=loss/2)
        print("\tDisc. Loss: {:.3f}".format(loss/2))
    print("===== End of Pretraining =====")
        
        
def train(args, G ,D, GAN, x_train, x_test, y_test, x_val, y_val, ano_data):
    ###Adversarial Training
    epochs = args.epochs
    batch_size = args.batch_size
    v_freq= args.v_freq
    ano_class = args.ano_class
    dataset = args.dataset
    
    if dataset == 'mnist':
        #Creating the result folder
        if not os.path.exists('./result/{}/'.format(args.dataset)):
            os.makedirs('./result/{}/'.format(args.dataset))
        result_path = './result/{}/{}'.format(args.dataset,len(os.listdir('./result/{}/'.format(args.dataset))))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists('{}/pictures'.format(result_path)):
            os.makedirs('{}/pictures'.format(result_path))
        if not os.path.exists('{}/histogram'.format(result_path)):
            os.makedirs('{}/histogram'.format(result_path))
            
        d_loss = []
        g_loss = []
        val_prc_list = []
        best_prc = 0
        best_test_prc = 0
        for epoch in range(epochs):
            try:
                with trange(x_train.shape[0]//batch_size, ascii=True, desc='Epoch {}'.format(epoch+1)) as t:
                    for step in t:
                        ###Train Discriminator
                        loss_temp = []
                        
                        set_trainability(D, True)
                        K.set_value(gw, [1])
                        x,y = D_data(batch_size,G,'real',x_train)
                        loss_temp.append(D.train_on_batch(x, y))
                        
                        set_trainability(D, True)
                        K.set_value(gw, [args.gamma])
                        x,y = D_data(batch_size,G,'gen',x_train)
                        loss_temp.append(D.train_on_batch(x,y))
                        
                        d_loss.append(sum(loss_temp)/len(loss_temp))
                        
                        ###Train Generator
                        set_trainability(D, False)
                        x = noise_data(batch_size)
                        y = np.zeros(batch_size)
                        y[:] = args.alpha
                        g_loss.append(GAN.train_on_batch(x,y))
                        
                        t.set_postfix(G_loss=g_loss[-1], D_loss=d_loss[-1])
            except KeyboardInterrupt: #hit control-C to exit and save video there
                break    
            
            show_images(G.predict(noise_data(16)),epoch+1,result_path)
            if (epoch + 1) % v_freq == 0:
                val_prc, test_prc = D_test(D, G, GAN, epoch, v_freq, x_val, y_val, x_test, y_test, ano_data, ano_class,result_path)
                val_prc_list.append(val_prc)
                
                f = open('{}/logs.txt'.format(result_path),'a+')
                f.write('\nEpoch: {}\n\tval_prc: {:.3f} \n\ttest_prc: {:.3f}'.format(epoch+1, val_prc, test_prc))
                f.close()
                
                if val_prc > best_prc:
                    best_prc = val_prc
                    best_test_prc = test_prc
                    G.save('{}/gen_{}.h5'.format(result_path,ano_class))
                    D.save('{}/dis_{}.h5'.format(result_path,ano_class))
                    
                print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}\n\tArea_prc: {:.3f}".format(g_loss[-1], d_loss[-1], val_prc_list[-1]))
                
                plt.figure()
                plt.plot([i*v_freq for i in range(len(val_prc_list))], val_prc_list, '-b', label='PRC')
                plt.legend()
                plt.title('AUPRC vs AUROC of ano_class {}'.format(ano_class))
                plt.xlabel('Epochs')
                plt.ylabel('%Area')
                plt.savefig('{}/area.png'.format(result_path), dpi=60)
                plt.close()
            else:
                print("\tGen. Loss: {:.3f}\n\tDisc. Loss: {:.3f}".format(g_loss[-1], d_loss[-1]))
        
        #Saving result in result.json file    
        result =[("best_test_prc",round(best_test_prc,3)),("val_prc",round(best_prc,3))]
        result_dict = OrderedDict(result)
        with open('{}/result.json'.format(result_path),'w+') as outfile:
            json.dump(result_dict, outfile, indent=4)

    elif dataset == 'cifar10':
        #Creating the result folder
        if not os.path.exists('./result/{}/'.format(args.dataset)):
            os.makedirs('./result/{}/'.format(args.dataset))
        result_path = './result/{}/{}'.format(args.dataset,len(os.listdir('./result/{}/'.format(args.dataset))))
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        if not os.path.exists('{}/pictures'.format(result_path)):
            os.makedirs('{}/pictures'.format(result_path))
        if not os.path.exists('{}/histogram'.format(result_path)):
            os.makedirs('{}/histogram'.format(result_path))
            
        d_loss = []
        g_loss = []
        val_roc_list = []
        best_roc = 0
        best_test_roc = 0
        
        for epoch in range(epochs):
            try:
                with trange(x_train.shape[0]//batch_size, ascii=True, desc='Epoch {}'.format(epoch+1)) as t:
                    for step in t:
                        ###Train Discriminator
                        loss_temp = []
                        
                        set_trainability(D, True)
                        K.set_value(gw, [1])
                        x,y = D_data(batch_size,G,'real',x_train, dataset = dataset)
                        loss_temp.append(D.train_on_batch(x, y))
                        
                        set_trainability(D, True)
                        K.set_value(gw, [args.gamma])
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
                        
                if (epoch + 1 % v_freq == 0):
                    print("Epoch #{}/{}: Discriminator Loss: {}, Generator Loss: {}".format(epoch, epochs,
                              d_loss[-1], g_loss[-1]))
                    # Save Generated Images
                    generated_images = G.predict(noise_data(50, dataset = 'cifar10'))
                    show_images(generated_images,epoch,result_path, dataset = 'cifar10')
    
                pred_test = D.predict(x_test)
                pred_test = np.reshape(pred_test, x_test.shape[0])
                pred_val = D.predict(x_val)
                pred_val = np.reshape(pred_val, x_val.shape[0])
                
                val_roc_list.append(roc_auc_score(y_val, pred_val))
                test_roc = roc_auc_score(y_test, pred_test)
                

                if (val_roc_list[-1] >= best_roc):
                    best_roc = val_roc_list
                    best_test_roc = test_roc
                    
                    G.save('{}/gen_{}.h5'.format(result_path,ano_class))
                    D.save('{}/dis_{}.h5'.format(result_path,ano_class))
                    
                    #Saving result in result.json file    
                    result =[("best_test_roc",round(best_test_roc,3)),("val_roc",round(best_roc,3))]
                    result_dict = OrderedDict(result)
                    with open('{}/result.json'.format(result_path),'w+') as outfile:
                        json.dump(result_dict, outfile, indent=4)
                        
            except KeyboardInterrupt: #hit control-C to exit and save video there
                break

def training_pipeline(args):
    seed(args.seed)
    set_random_seed(args.seed)
    if args.dataset == 'mnist':
        x_train, x_test, y_test, x_val, y_val, ano_data = load_data(args)
    elif args.dataset == 'cifar10':
        x_train, x_test, y_test, x_val, y_val = load_data(args)
        ano_data = 0
    G, D, GAN = load_model(args)
    pretrain(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val, ano_data, dataset = args.dataset)
    train(args, G, D, GAN, x_train, x_test, y_test, x_val, y_val, ano_data)