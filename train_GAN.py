# -*- coding: utf-8 -*- # @Author: aksharasoman # @Date: 2024-07-03 14:19:37 # @Last Modified by: aksharasoman # @Last Modified time: 2024-07-03 15:15:43import torch
import torch.nn as nn
from config import *
from torchvision.utils import save_image
import torch
# Fake loss
def get_fake_loss(disc_out):
  criterion = nn.BCEWithLogitsLoss()
  ground_truth = torch.zeros_like(disc_out)
  return criterion(disc_out,ground_truth)

# real loss
def get_real_loss(disc_out):
  criterion = nn.BCEWithLogitsLoss()
  ground_truth = torch.ones_like(disc_out)
  return criterion(disc_out,ground_truth)

def train_GAN(trainloader,D,G,D_opt,G_opt):
    for i in range(epochs):
    # for each epoch:
      total_d_loss = 0.0
      total_g_loss = 0.0
      for real_img, _ in (trainloader): #tqdm is used to show the progress bar for loop execution
        # for each mini-batch:
        real_img = real_img.to(device) # move to GPU

        # Train Discriminator
        D_opt.zero_grad() # set D gradients to 0

        random_noise_vec = torch.randn(batch_size,noise_dim,device=device)
        gen_img = G(random_noise_vec)
        gen_Dout = D(gen_img)
        fake_loss = get_fake_loss(gen_Dout)

        real_Dout = D(real_img)
        real_loss = get_real_loss(real_Dout)

        d_loss = (real_loss + fake_loss)/2
        total_d_loss += d_loss.item()
        d_loss.backward()
        D_opt.step()

        # Train Generator
        G_opt.zero_grad() # set G gradients to zero

        random_noise_vec = torch.randn(batch_size,noise_dim,device=device)
        gen_img = G(random_noise_vec)
        gen_Dout = D(gen_img)
        g_loss = get_real_loss(gen_Dout)

        total_g_loss += g_loss.item()
        g_loss.backward()
        G_opt.step()

      avg_d_loss = total_d_loss/len(trainloader) # computed after each epoch
      avg_g_loss = total_g_loss/len(trainloader)

      print('Epoch: {} | D_loss: {} | G_loss: {}'.format(i+1,avg_d_loss,avg_g_loss))
      fname = f'results/generated_digits_epoch{i+1}.png'
      save_image(gen_img[:36],fname,nrow=6)
    
    # save model state  dictionary
    torch.save(D.state_dict(), 'discriminator_state.pth')
    torch.save(G.state_dict(), 'generator_state.pth')


    
