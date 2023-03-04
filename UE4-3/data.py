import argparse
import numpy as np
import torch
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import yaml

CFG = {}
with open("config.yaml") as f:
    CFG = yaml.safe_load(f)
    CFG = argparse.Namespace(**CFG)

def load_dataset():
    dataset = dset.ImageFolder(root=CFG.dataroot,
                               transform=transforms.Compose([
                                   transforms.Resize((CFG.image_size, CFG.image_size)),
                                   transforms.CenterCrop(CFG.image_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(*CFG.stats),
                               ]))
    
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    np.random.shuffle(indices)
    val_split_index = int(np.floor(0.2 * dataset_size))
    val_idx, train_idx = indices[:val_split_index], indices[val_split_index:]
    
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=CFG.batch_size,
                                             shuffle=False,
                                             num_workers=CFG.n_workers,
                                            sampler=SubsetRandomSampler(train_idx))
    
    val_dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=CFG.batch_size,
                                             shuffle=False,
                                             num_workers=CFG.n_workers,
                                            sampler=SubsetRandomSampler(val_idx))
    print("datasets loaded")
    return train_dataloader, val_dataloader
