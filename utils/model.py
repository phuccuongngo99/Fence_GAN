from utils.custom_losses import *

import tensorflow as tf

import keras.backend as K
from keras import losses
from keras.models import Model
from keras.layers import Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU
from keras.layers import Input, Dense, BatchNormalization, Activation, Dropout
from keras.regularizers import l2
from keras.optimizers import Adam

init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)

gamma = K.variable([1])

def load_model(args):
    if args.dataset == 'mnist': 
        return get_mnist_model(args)
    if args.dataset == 'cifar10':
        return get_cifar10_model(args)
    
def set_trainability(model, trainable=False): #alternate to freeze D network while training only G in (G+D) combination
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable        

def D_loss(y_true, y_pred):
    loss_gen = losses.binary_crossentropy(y_true,y_pred)
    loss = gamma * loss_gen
    return loss

def get_cifar10_model(args):
    '''
    Return: G, D, GAN models
    '''
    
    '''
    Build Generator
    '''
    G_in = Input(shape=(256,))
    x = Dense(256 * 2 * 2, kernel_initializer = 'glorot_normal')(G_in)
    x = Reshape((2, 2, 256))(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(128, (5,5), padding = 'same', kernel_initializer = 'glorot_normal', strides = 2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(64, (5,5), padding = 'same', kernel_initializer = 'glorot_normal', strides = 2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(32, (5,5), padding = 'same', kernel_initializer = 'glorot_normal', strides = 2)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2DTranspose(3, (5,5), padding = 'same', kernel_initializer = 'glorot_normal', strides = 2)(x)
    G_out = Activation('tanh')(x)
    G = Model(G_in, G_out)
    
    '''
    Build Discriminator
    '''
    D_in = Input(shape = (32, 32, 3))
    x = Conv2D(32, (5,5), strides = 2, kernel_initializer = 'glorot_normal', kernel_regularizer = l2(args.d_l2), padding = 'same')(D_in)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(64, (5,5), strides = 2, kernel_initializer = 'glorot_normal', kernel_regularizer = l2(args.d_l2), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(128, (5,5), strides = 2, kernel_initializer = 'glorot_normal', kernel_regularizer = l2(args.d_l2), padding = 'same')(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(256, (5,5), strides = 2, kernel_initializer = 'glorot_normal', kernel_regularizer = l2(args.d_l2), padding = 'same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Flatten()(x)
    x = Dropout(0.2)(x)
    D_out = Dense(1, kernel_initializer = 'glorot_normal', activation = 'sigmoid')(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr = args.d_lr, beta_1 = 0.5, beta_2 = 0.999, decay = 1e-5)
    D.compile(loss = D_loss, optimizer = dopt)
    
    '''
    Building GAN
    '''
    set_trainability(D, False)
    GAN_in = Input(shape=(256,))
    G_out = G(GAN_in)
    GAN_out = D(G_out)
    GAN = Model(GAN_in, GAN_out)
    gopt = Adam(lr = args.g_lr, beta_1=0.5, beta_2=0.999)
    GAN.compile(loss = com_conv(G_out,args.beta,2), optimizer=gopt)
    
    return G, D, GAN

def get_mnist_model(args):
    '''
    Return: G, D, GAN
    '''
    
    '''
    Building Generator
    '''
    Z_in = Input(shape=(200,))
    x = Dense(1024, activation = 'relu', kernel_initializer=init_kernel)(Z_in)
    x = BatchNormalization()(x)
    
    x = Dense(7*7*128, activation = 'relu', kernel_initializer=init_kernel)(x)
    x = BatchNormalization()(x)
    
    x = Reshape((7,7,128))(x)
    x = Conv2DTranspose(64, (4,4), strides=(2, 2), padding='same', activation='relu', kernel_initializer=init_kernel)(x)
    x = BatchNormalization()(x)
    
    G_out = Conv2DTranspose(1, (4,4), strides=(2, 2), padding='same', activation='tanh', kernel_initializer=init_kernel)(x)
    G = Model(Z_in, G_out)
    
    '''
    Builiding Discriminator
    '''
    D_in = Input(shape=(28,28,1))
    x = Conv2D(64, (4,4), strides=(2, 2), padding='same', kernel_initializer=init_kernel)(D_in)
    x = LeakyReLU(0.1)(x)
    
    x = Conv2D(64, (4,4), strides=(2, 2), padding='same', kernel_initializer=init_kernel)(x)
    x = LeakyReLU(0.1)(x)
    
    x = Flatten()(x)
    x = Dense(1024, kernel_initializer=init_kernel)(x)
    x = LeakyReLU(0.1)(x)
    
    D_out = Dense(1, activation='sigmoid', kernel_initializer=init_kernel)(x)
    D = Model(D_in, D_out)
    dopt = Adam(lr=args.d_lr,beta_1=0.5, beta_2=0.999)
    gamma = K.variable([1])
    D.compile(loss=D_loss,optimizer=dopt)

    '''
    Building GAN
    '''
    set_trainability(D, False)
    GAN_in = Input(shape=(200,))
    G_out = G(GAN_in)
    GAN_out = D(G_out)
    GAN = Model(GAN_in, GAN_out)
    gopt = Adam(lr=args.g_lr, beta_1=0.5, beta_2=0.999)
    GAN.compile(loss=com_conv(G_out,args.beta,2), optimizer=gopt)
    
    return G, D, GAN