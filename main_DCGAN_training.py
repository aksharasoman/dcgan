# -*- coding: utf-8 -*- # @Author: aksharasoman # @Date: 2024-07-03 13:49:04 # @Last Modified by: aksharasoman # @Last Modified time: 2024-07-03 13:49:05
'''
Project Title: Generate Handwritten Digits using DCGAN
More details in obsidian
'''
import torch
import matplotlib.pyplot as plt
from config import * #configuration values
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader
from discriminator import Discriminator
from generator import Generator
from train_GAN import train_GAN
import torch.nn as nn
torch.manual_seed(32)

#Replace random initialized weights to Normal weights
def weights_init(m):
  if isinstance(m,nn.Conv2d) or isinstance(m,nn.ConvTranspose2d):
    nn.init.normal_(m.weight,0.0,0.02)
  if isinstance(m,nn.BatchNorm2d):
    nn.init.normal_(m.weight,0.0,0.02)
    nn.init.constant_(m.bias,0) # all set to 0

def main(): 
    
    # Load Dataset
    train_aug = T.Compose([
        T.ToTensor(),
        T.RandomRotation((-20,20))
    ])
    
    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=train_aug)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Model 
    G = Generator(noise_dim).to(device)
    D = Discriminator().to(device)
    
    # Intialize model with normal distributed weights
    G = G.apply(weights_init)
    D = D.apply(weights_init)
    
    # Optimizer
    G_opt = torch.optim.Adam(G.parameters(),lr=lr,betas=(beta_1,beta_2))
    D_opt = torch.optim.Adam(D.parameters(),lr=lr,betas=(beta_1,beta_2))
        
    # Training
    train_GAN(trainloader,D,G,D_opt,G_opt)


torch.manual_seed(32)
    
if __name__ == "__main__": 
    main()