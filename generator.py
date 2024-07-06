# -*- coding: utf-8 -*- # @Author: aksharasoman # @Date: 2024-07-03 14:06:25 # @Last Modified by: aksharasoman # @Last Modified time: 2024-07-03 14:06:26import torch.nn as nn 
import torch.nn as nn

def get_generator_block(in_channels,out_channels,kernel_size,stride,final_block = False):
  if final_block:
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride),
        nn.Tanh()
    )
  return nn.Sequential(
      nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride),
      nn.BatchNorm2d(out_channels),
      nn.ReLU()
  )
  
class Generator(nn.Module):
  def __init__(self,z_dim):
    super().__init__()
    self.z_dim = z_dim

    self.block1 = get_generator_block(z_dim,256,(3,3),2)
    self.block2 = get_generator_block(256,128,(4,4),1)
    self.block3 = get_generator_block(128,64,(3,3),2)

    self.block4 = get_generator_block(64,1,(4,4),2,final_block=True)

  def forward(self,r_noise_vec):
    # (bs,noise_dim) -> (bs,noise_dim,1)
    x = r_noise_vec.view(-1,self.z_dim,1,1) #reshape random noise vector
    x = self.block1(x)
    x = self.block2(x)
    x = self.block3(x)
    x = self.block4(x)
    return x
  
'''

Network : Generator

z_dim = 64
input : (bs,z_dim)

      |
      | Reshape
      V

input : (bs, channel, height, width) -> (bs, z_dim , 1 , 1)
      |                                                                                               ---- SUMMARY ----
      V
ConvTranspose2d( in_channels = z_dim, out_channels = 256, kernel_size = (3,3), stride = 2)             #(bs, 256, 3, 3)
BatchNorm2d()                                                                                          #(bs, 256, 3, 3)
ReLU()                                                                                                 #(bs, 256, 3, 3)
      |
      V
ConvTranspose2d( in_channels = 256, out_channels = 128, kernel_size = (4,4), stride = 1)               #(bs, 128, 6, 6)
BatchNorm2d()                                                                                          #(bs, 128, 6, 6)
ReLU()                                                                                                 #(bs, 128, 6, 6)
      |
      V
ConvTranspose2d( in_channels = 128, out_channels = 64, kernel_size = (3,3), stride = 2)                #(bs, 64, 13, 13)
BatchNorm2d()                                                                                          #(bs, 64, 13, 13)
ReLU()                                                                                                 #(bs, 64, 13, 13)
      |
      V
ConvTranspose2d( in_channels = 64, out_channels = 1, kernel_size = (4,4), stride = 2)                  #(bs, 1, 28, 28)
Tanh()                                                                                                 #(bs, 1, 28, 28)

'''