import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve, average_precision_score
from PIL import Image

def noise_data(n):
    return np.random.normal(0,1,[n,200])

def deprocess(x, dataset = 'mnist'):
    if dataset == 'mnist':
        x = (x+1)/2 * 255
        x = np.clip(x, 0, 255)
        x = np.uint8(x)
        x = x.reshape(-1, 28, 28)
        return x
    elif dataset == 'cifar10':
        x = x * 127.5 + 127.5
        return x

def show_images(img_array,epoch,result_path, dataset = 'mnist'):
    if dataset == 'mnist':
        n_images = deprocess(img_array[:16])
        rows = 4
        cols = 4
        
        plt.figure(figsize=(cols, rows))
        for i in range(len(n_images)):
            img = n_images[i,...]
            plt.subplot(rows, cols, i+1)
            plt.imshow(img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        if not os.path.exists('{}/pictures'.format(result_path)):
            os.makedirs('{}/pictures'.format(result_path))
        plt.savefig('{}/pictures/generated_{}.png'.format(result_path,epoch))
        plt.close()
    
    elif dataset == 'cifar10': 
        total,width,height, channel = img_array.shape[:]
        img_array = deprocess(img_array)
        cols = int(math.sqrt(total))
        rows = math.ceil(float(total)/cols)
        combined_image = np.zeros((int(height*rows), int(width*cols), int(channel)),
                                  dtype=img_array.dtype)
    
        for index, image in enumerate(img_array):
            i = int(index/cols)
            j = index % cols
            combined_image[width*i:width*(i+1), height*j:height*(j+1), 0:3] = image[:, :, :]
        if not os.path.exists('{}/pictures'.format(result_path)):
            os.makedirs('{}/pictures'.format(result_path))
        img = Image.fromarray(combined_image.astype(np.uint8), 'RGB')
        img.save('{}/pictures/generated_{}.png'.format(result_path,epoch))
        img.show()
    

def D_test(D, G, GAN, epoch, v_freq, x_val, y_val, x_test, y_test, ano_class,result_path):
    ###Plotting AUPRC curve
    ###Histogram of predicted scores of nornmal data, anomalous data and generated data
    
    ###VALIDATION
    y_true = y_val
    y_pred = np.squeeze(D.predict(x_val))

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    val_prc = auc(recall, precision)
    
    ###TEST
    y_true = y_test
    y_pred = np.squeeze(D.predict(x_test))

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred)
    test_prc = auc(recall, precision)
    
    return val_prc, test_prc


def histogram(G, D, GAN, x_test, y_test, result_path):
    y_gen_pred = np.squeeze(GAN.predict(noise_data(5000)))
    y_pred = np.squeeze(D.predict(x_test))
    
    plt.figure()
    #1 is normal data
    #0 is anomalous data
    plt.hist(y_pred[np.where(y_test==0)], density=True, bins=100, range=(0,1.0), label='anomalous', color='r', alpha=0.5)
    plt.hist(y_pred[np.where(y_test==1)], density=True, bins=100, range=(0,1.0), label='normal', color='b', alpha=0.5)
    plt.hist(y_gen_pred, density=True, bins=100, range=(0,1.0), label='generated', color='g', alpha=0.5)
    
    #plt.axis([0, 1, 0, 1]) 
    plt.xlabel('Confidence')
    plt.ylabel('Probability')
    plt.title('AUPRC on test set', fontsize=20)
    plt.legend(loc=9)
    plt.savefig('{}/histogram/histogram_at_best_epoch.png'.format(result_path),dpi=60)
    plt.close()