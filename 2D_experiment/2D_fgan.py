import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Add
from keras.layers.core import Dense
from keras.optimizers import Adam
from keras import losses
import keras.backend as K
from custom_losses import com

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(1)

###Training hyperparameters###
epoch = 30001
batch_size = 100


###Generator Hyperparameters###
alpha = 0.5
beta = 15

###Discriminator Hyperparameters###
gamma = 0.1


gm = K.variable([1])

if not os.path.exists('./pictures'):
    os.makedirs('./pictures')

def animate(G,D,epoch,v_animate):
    plt.figure()
    xlist = np.linspace(0, 40, 40)
    ylist = np.linspace(0, 40, 40)
    X, Y = np.meshgrid(xlist, ylist)
    In = np.array(np.meshgrid(xlist,ylist)).T.reshape(-1,2)
    Out = D.predict(In)
    Z = Out.reshape(40,40).T
    c = ('#66B2FF','#99CCFF','#CCE5FF','#FFCCCC','#FF9999','#FF6666')
    cp = plt.contourf(X, Y, Z,[0.0,0.2,0.4,0.5,0.6,0.8,1.0],colors=c)
    plt.colorbar(cp)
 
    rx,ry = data_D(G,500,'real')[0].T
    gx,gy = data_D(G,200,'gen')[0].T
    #plotting the sample data, generated data
    plt.scatter(rx,ry,color='red')
    plt.scatter(gx,gy,color='blue')
    plt.xlabel('x-axis')
    plt.xlim(0,40)
    plt.ylabel('y-axis')
    plt.ylim(0,40)
    plt.title('Epoch'+str(epoch))
    plt.savefig('./pictures/'+str(int(epoch/v_animate))+'.png', dpi=500)
    plt.close()

def real_data(n):
    return np.random.normal((20,20),3,[n,2])

def noise_data(n):
    return np.random.normal(0,8,[n,2])
    
#Prepare training dataset for Discriminator
def data_D(G,n_samples,mode):
    if mode == 'real':
        x = real_data(n_samples)
        y = np.ones(n_samples)
        return x, y
        
    elif mode == 'gen':
        x = G.predict(noise_data(n_samples))
        y = np.zeros(n_samples)
        return x, y
 
#Prepare training dataset for Generator
def data_G(batch_size):
    x = noise_data(batch_size)
    y = np.zeros(batch_size)
    y[:] = alpha
    return x, y

#Discriminator Loss function
def D_loss(y_true, y_pred):
    loss_gen = losses.binary_crossentropy(y_true,y_pred)
    loss = gm*loss_gen
    return loss

#Generator model
def get_generative():
    G_in = Input(shape=(2,))
    x = Dense(10, activation='relu')(G_in)
    x = Dense(10, activation='relu')(x)
    #G_out = Dense(2)(x)
    x = Dense(2)(x)
    G_out = Add()([G_in,x])
    G = Model(G_in, G_out)
    return G

#Discriminator model
def get_discriminative():
    D_in = Input(shape=(2,))
    x = Dense(15, activation='relu')(D_in)
    x = Dense(15, activation='relu')(x)
    D_out = Dense(1, activation='sigmoid')(x)
    D = Model(D_in, D_out)
    D.compile(loss=D_loss, optimizer=Adam(lr=1e-3))
    return D

def set_trainability(model, trainable=False):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable
    
def make_gan(G, D):  #making (G+D) framework
    set_trainability(D, False)
    GAN_in = Input(shape=(2,))
    G_out = G(GAN_in)
    GAN_out = D(G_out)
    GAN = Model(GAN_in, GAN_out)
    GAN.compile(loss=com(G_out,beta,2), optimizer=Adam(lr=1e-3))
    return GAN

G = get_generative()
D = get_discriminative()
GAN = make_gan(G,D)

def pretrain(G, D, n_samples=batch_size): #pretrain D
    for epoch in range(20):
        loss_temp = []
        set_trainability(D, True)
        K.set_value(gm, [1])
        x,y = data_D(G, n_samples, 'real')
        loss_temp.append(D.train_on_batch(x, y))
        
        set_trainability(D, True)
        K.set_value(gm, [gamma])
        x,y = data_D(G, n_samples, 'gen')
        loss_temp.append(D.train_on_batch(x,y))
        
        print('Pretrain Epoch {} Dis Loss {}'.format(epoch, sum(loss_temp)/len(loss_temp)))
        
pretrain(G, D)

def train(GAN, G, D, epochs=epoch, n_samples=batch_size, v_freq=100, v_animate=1000):
    d_loss = []
    g_loss = []
#    data_show = sample_noise(n_samples=n_samples)[0]
    for epoch in range(epochs):
        try:
            loss_temp = []
            set_trainability(D, True)
            K.set_value(gm, [1])
            x,y = data_D(G, n_samples, 'real')
            loss_temp.append(D.train_on_batch(x, y))
            
            set_trainability(D, True)
            K.set_value(gm, [gamma])
            x,y = data_D(G, n_samples, 'gen')
            loss_temp.append(D.train_on_batch(x,y))
            
            d_loss.append(sum(loss_temp)/len(loss_temp))
            
            ###Train Generator
            X, y = data_G(n_samples)
            set_trainability(D, False)
            generator_loss = GAN.train_on_batch(X,y)
            g_loss.append(generator_loss)
            
            if (epoch + 1) % v_freq == 0:
                print("Epoch #{}: Generative Loss: {}, Discriminative Loss: {}".format(epoch + 1, g_loss[-1], d_loss[-1]))
            if epoch % v_animate == 0:
                animate(G,D,epoch,v_animate)
        except KeyboardInterrupt:
            break
    G.save('Circle_G.h5')
    D.save('Circle_D.h5')   
    return d_loss, g_loss

d_loss, g_loss = train(GAN, G, D)
