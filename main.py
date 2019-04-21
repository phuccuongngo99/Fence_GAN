import argparse
from fgan_train import training_pipeline

parser = argparse.ArgumentParser('Train your Fence GAN')

###Training hyperparameter
args = parser.add_argument('--dataset', default='mnist', help='2d | kdd99 | mnist | cifar10')
args = parser.add_argument('--ano_class',type=int, default=0, help='1 anomaly class')
args = parser.add_argument('--epochs', type=int, default=2, help='number of epochs to train')

###FenceGAN hyperparameter
args = parser.add_argument('--beta',type=int, default=30,help='beta')
args = parser.add_argument('--gamma',type=int, default=0.1, help='gamma')
args = parser.add_argument('--alpha',type=int, default=0.1, help='alpha')

###Other hyperparameters
args = parser.add_argument('--batch_size',type=int, default=200, help='')
args = parser.add_argument('--pretrain',type=int, default=0, help='number of pretrain epoch')
args = parser.add_argument('--d_lr', default=1e-5, help='learning_rate of discriminator')
args = parser.add_argument('--g_lr', default=2e-5, help='learning rate of generator')
args = parser.add_argument('--v_freq', type=int, default=1, help='epoch frequency to evaluate performance')

args = parser.parse_args()

training_pipeline(args)