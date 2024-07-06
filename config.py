# -*- coding: utf-8 -*- # @Author: aksharasoman # @Date: 2024-07-03 14:01:07 # @Last Modified by: aksharasoman # @Last Modified time: 2024-07-03 14:01:08# config.py

device = 'cuda'
batch_size = 128

noise_dim = 64 #shape of the random noise vector in the generator

# Optimizer parameters
lr = 0.002
beta_1 = 0.5
beta_2 = 0.999

# Training Parameters
epochs = 20