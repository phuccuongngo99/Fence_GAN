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
        

def get_mnist(ano_class):
    (x_tr, y_tr), (x_tst, y_tst) = mnist.load_data()
    
    x_total = np.concatenate([x_tr, x_tst])
    y_total = np.concatenate([y_tr, y_tst])
    print(x_total.shape, y_total.shape)
    
    x_total = x_total.reshape(-1, 28, 28, 1)
    x_total = preprocess(x_total)
    
    ###
    #There are 2 classes: normal and anomalous
    #Training data: x_train: 80% of data/images from normal classes
    #Validation data: x_val: 5% of data/images from normal classes + 25% of data/images from anomalous classes
    #Testing data: x_test: 15% of data/images from normal classes + 75% of data/images from anomalous classes
    ###
    
    delete = []
    for count, i in enumerate(y_total):
        if i == ano_class:
            delete.append(count)
    
    ano_data = x_total[delete,:]
    normal_data = np.delete(x_total,delete,axis=0)
    
    normal_num = normal_data.shape[0] #Number of data/images of normal classes
    ano_num = ano_data.shape[0] #Number of data/images of anomalous classes
    
    del x_total
    
    x_train = normal_data[:int(0.8*normal_num),...]
    
    x_test = np.concatenate((normal_data[int(0.8*normal_num):int(0.95*normal_num),...], ano_data[:ano_num*3//4]))
    y_test = np.concatenate((np.ones(int(0.95*normal_num)-int(0.8*normal_num)), np.zeros(ano_num*3//4)))
    
    x_val = np.concatenate((normal_data[int(0.95*normal_num):,...], ano_data[ano_num*3//4:]))
    y_val = np.concatenate((np.ones(normal_num-int(0.95*normal_num)), np.zeros(ano_num-ano_num*3//4)))
    
    return x_train, x_test, y_test, x_val, y_val, ano_data

#def get_cifar10():

#def get_kdd99():
    
#def get_custom():