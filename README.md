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
2. Installing tensorflow or tensorflow-gpu by following instruction at (https://www.tensorflow.org/install/pip)

3. Installing necessary libraries
    ```
    pip3 install -r requirements.txt
    ```

## Anomaly Detection

### 2D Synthetic Dataset

### MNIST
    python3 --dataset mnist --ano_class 0 --epochs 100
    
Check results and plots under `result` folder
