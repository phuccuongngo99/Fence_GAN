import os
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.metrics import auc, precision_recall_curve, roc_curve
from PIL import Image

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

def show_images(img_array,result_path, dataset = 'mnist'):
    if dataset == 'mnist':
        n_images = deprocess(img_array[:16])
        
        plt.figure(figsize=(4,4))
        for i in range(len(n_images)):
            img = n_images[i,...]
            plt.subplot(4, 4, i+1)
            plt.imshow(img, cmap='gray')
            plt.xticks([])
            plt.yticks([])
        plt.tight_layout()
        plt.savefig('{}/generated_data_at_best_epoch.png'.format(result_path))
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
        img.save('{}/pictures/generated.png'.format(result_path))
        img.show()
    

def compute_au(D, G, GAN, x_val, y_val, x_test, y_test, mode):
    '''
    Return auprc or auroc evaluated on validation/test set
    '''
    if mode == 'auprc':
        ###VALIDATION
        y_pred_val = np.squeeze(D.predict(x_val))
        precision, recall, _ = precision_recall_curve(y_val, y_pred_val)
        val_prc = auc(recall, precision)
        
        ###TEST
        y_pred_test = np.squeeze(D.predict(x_test))
        precision, recall, _ = precision_recall_curve(y_test, y_pred_test)
        test_prc = auc(recall, precision)
    
        return val_prc, test_prc
    
    elif mode == 'auroc':
        ###VALIDATION
        y_pred_val = np.squeeze(D.predict(x_val))
        fpr, tpr, _ = roc_curve(y_val, y_pred_val)
        val_roc = auc(fpr, tpr)
        
        ###TEST
        y_pred_val = np.squeeze(D.predict(x_val))
        fpr, tpr, _ = roc_curve(y_val, y_pred_val)
        val_roc = auc(fpr, tpr)
        
        return val_roc, val_prc
        

def histogram(G, D, GAN, x_test, y_test, result_path, latent_dim):
    y_gen_pred = np.squeeze(GAN.predict(np.random.normal(0,1,[5000,latent_dim])))
    y_pred = np.squeeze(D.predict(x_test))
    
    plt.figure()
    #1 is normal data
    #0 is anomalous data
    plt.hist(y_pred[np.where(y_test==0)], density=True, bins=100, range=(0,1.0), label='anomalous', color='r', alpha=0.5)
    plt.hist(y_pred[np.where(y_test==1)], density=True, bins=100, range=(0,1.0), label='normal', color='b', alpha=0.5)
    plt.hist(y_gen_pred, density=True, bins=100, range=(0,1.0), label='generated', color='g', alpha=0.5)
    
    #plt.axis([0, 1, 0, 1]) 
    plt.xlabel('Discriminator Score')
    plt.ylabel('Frequency of Occurence')
    plt.title('Histogram of Discriminator Score at the Best Epoch')
    plt.legend(loc=9)
    plt.savefig('{}/histogram_at_best_epoch.png'.format(result_path),dpi=60)
    plt.close()