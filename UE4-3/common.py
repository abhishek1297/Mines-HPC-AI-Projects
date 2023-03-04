import os
from tqdm import tqdm
import argparse
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import numpy as np
from torchvision.utils import save_image, make_grid
from torch.utils.tensorboard import SummaryWriter
from data import load_dataset
from datetime import timedelta
from time import time

import yaml
MODEL_DIR = "models"
GEN_IMG_DIR = "gen-images"
CFG = {}
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)
    CFG = argparse.Namespace(**CFG)

# Establish convention for real and fake labels during training
real_label = 1.
fake_label = 0.

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def denorm(img_tensors):
    return img_tensors * CFG.stats[1][0] + CFG.stats[0][0]
        
def display_samples(generator, index, latent_tensors, save=True, show=False):
    
    import matplotlib.pyplot as plt
    
    fake_images = generator(latent_tensors)
    fake_fname = 'gen-{0:0=4d}.png'.format(index)
    if save:
        save_image(denorm(fake_images), os.path.join("gen-imgs", fake_fname), nrow=8)
    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8, normalize=True, padding=2).permute(1, 2, 0))