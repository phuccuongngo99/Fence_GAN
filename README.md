# Fence GAN: Toward Better Anomaly Detection

This is the official implementation of the paper: Fence GAN: Towards Better Anomaly Detection [(link)](https://arxiv.org/abs/1904.01209).

## Prerequisites
1. Linux OS
2. Python 3
3. CUDA 

## Installation
1. Clone repository
    ```
    git clone https://github.com/phuccuongngo99/Fence_GAN.git
    ```
2. Installing tensorflow or tensorflow-gpu by following instruction [here](https://www.tensorflow.org/install/pip).

3. Installing necessary libraries
    ```
    pip3 install -r requirements.txt
    ```

## Anomaly Detection

### 2D Synthetic Dataset
    
    python3 2D_experiment/2D_fgan.py
    
### MNIST
    python3 main.py --dataset mnist --ano_class 0 --epochs 100 --alpha 0.1 --beta 30 --gamma 0.1 --batch_size 200 --pretrain 0 --d_lr 1e-5 --g_lr 2e-5 --v_freq 1 --latent_dim 200 --evaluation 'auprc'
Check results and plots under `result` folder


### CIFAR10
    python3 main.py --dataset cifar10 --ano_class 0 --epochs 150 --alpha 0.5 --beta 10 --gamma 0.5 --batch_size 128 --pretrain 15 --d_lr 1e-4 --g_lr 1e-3 --v-freq 1 --latent_dim 256 --evaluation 'auroc'
    
### Training option
Enter the commannd `python3 main.py -h` for help for more training options

```
usage: Train your Fence GAN [-h] [--dataset DATASET] [--ano_class ANO_CLASS]
                            [--epochs EPOCHS] [--beta BETA] [--gamma GAMMA]
                            [--alpha ALPHA] [--batch_size BATCH_SIZE]
                            [--pretrain PRETRAIN] [--d_l2 D_L2] [--d_lr D_LR]
                            [--g_lr G_LR] [--v_freq V_FREQ] [--seed SEED]
                            [--evaluation EVALUATION]
                            [--latent_dim LATENT_DIM]

optional arguments:
  -h, --help            show this help message and exit
  --dataset DATASET     mnist | cifar10
  --ano_class ANO_CLASS
                        1 anomaly class
  --epochs EPOCHS       number of epochs to train
  --beta BETA           beta
  --gamma GAMMA         gamma
  --alpha ALPHA         alpha
  --batch_size BATCH_SIZE
  --pretrain PRETRAIN   number of pretrain epoch
  --d_l2 D_L2           L2 Regularizer for Discriminator
  --d_lr D_LR           learning_rate of discriminator
  --g_lr G_LR           learning rate of generator
  --v_freq V_FREQ       epoch frequency to evaluate performance
  --seed SEED           numpy and tensorflow seed
  --evaluation EVALUATION
                        'auprc' or 'auroc'
  --latent_dim LATENT_DIM
                        Latent dimension of Gaussian noise input to Generator

```
## Citation
```
@article{fencegan2019,
  author    = {Cuong Phuc Ngo and Amadeus Aristo Winarto and Connie Khor Li Kou and 
               Sojeong Park and Farhan Akram and Hwee Kuan Lee},
  title     = {Fence GAN: Towards Better Anomaly Detection},
  year      = {2019},
  url       = {https://arxiv.org/abs/1904.01209},
  archivePrefix = {arXiv}
}
```
