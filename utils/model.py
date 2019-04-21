from utils.custom_losses import *
import keras.backend as K
import tensorflow as tf
from keras import losses
from keras.models import Model
from keras.layers import Flatten, Reshape, Conv2D, Conv2DTranspose, LeakyReLU, Input, Dense, BatchNormalization
from keras.optimizers import Adam

init_kernel = tf.random_normal_initializer(mean=0.0, stddev=0.02)
gm = K.variable([1])

def load_model(args):
    if args.dataset == 'mnist': 
        return get_mnist_model(args)
    
def set_trainability(model, trainable=False): #alternate to freeze D network while training only G in (G+D) combination
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable        

def D_loss(y_true, y_pred):
    loss_gen = losses.binary_crossentropy(y_true,y_pred)
    loss = gm*loss_gen
    return loss

def get_mnist_model(args):
###Return: G, D_r, D_g, GAN
    ###Making Generator
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
    
    ###Making Discriminator
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
    gm = K.variable([1])
    D.compile(loss=D_loss,optimizer=dopt)

    ###Making GAN
    set_trainability(D, False)
    GAN_in = Input(shape=(200,))
    G_out = G(GAN_in)
    GAN_out = D(G_out)
    GAN = Model(GAN_in, GAN_out)
    gopt = Adam(lr=args.g_lr, beta_1=0.5, beta_2=0.999)
    GAN.compile(loss=com_conv(G_out,args.beta,2), optimizer=gopt)
    
    return G, D, GAN