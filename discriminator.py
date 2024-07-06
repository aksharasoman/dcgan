# -*- coding: utf-8 -*- # @Author: aksharasoman # @Date: 2024-07-03 14:05:36 # @Last Modified by: aksharasoman # @Last Modified time: 2024-07-03 14:05:37
# Create Discriminator network (DCGAN)
import torch.nn as nn

def disc_block(in_channels, out_channels, kernel_size, stride=2):
  block = nn.Sequential(nn.Conv2d(in_channels, out_channels,kernel_size,stride),
                        nn.BatchNorm2d(out_channels),
                        nn.LeakyReLU(0.2))
  return block

class Discriminator(nn.Module):
  def __init__(self):
    super().__init__()

    self.block1 = disc_block(1,16,(3,3))
    self.block2 = disc_block(16,32,(5,5))
    self.block3 = disc_block(32,64,(5,5))

    self.flatten = nn.Flatten()
    self.linear = nn.Linear(64,1)

  def forward(self,x):
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.flatten(x)
    x = self.linear(x) # NB: we are not using sigmoid output layer: since BCEwithLogitsLoss() has sigmoid and BCE loss in a single layer which provides better stability.
    return x

'''
Network : Discriminator

input : (bs, 1, 28, 28)
      |                                                                                               ---- SUMMARY ----
      V
Conv2d( in_channels = 1, out_channels = 16, kernel_size = (3,3), stride = 2)                           #(bs, 16, 13, 13)
BatchNorm2d()                                                                                          #(bs, 16, 13, 13)
LeakyReLU()                                                                                            #(bs, 16, 13, 13)
      |
      V
Conv2d( in_channels = 16, out_channels = 32, kernel_size = (5,5), stride = 2)                          #(bs, 32, 5, 5)
BatchNorm2d()                                                                                          #(bs, 32, 5, 5)
LeakyReLU()                                                                                            #(bs, 32, 5, 5)
      |
      V
Conv2d( in_channels = 32, out_channels = 64, kernel_size = (5,5), stride = 2)                          #(bs, 64, 1, 1)
BatchNorm2d()                                                                                          #(bs, 64, 1, 1)
LeakyReLU()                                                                                            #(bs, 64, 1, 1)
      |
      V
Flatten()                                                                                              #(bs, 64)
Linear(in_features = 64, out_features = 1)                                                             #(bs, 1)

'''