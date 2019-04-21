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
    python3 --dataset mnist --ano_class 0 --epochs 100
    
Check results and plots under `result` folder


### CIFAR10
    python3 --dataset cifar10 --ano_class 0 --epochs 150 --beta 10 --alpha 0.5 --gamma 0.5 --d_lr 1e-4 --g_lr 1e-3 --v-freq 1 --pretrain 15
    
