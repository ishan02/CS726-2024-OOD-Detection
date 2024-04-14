import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm

from build_dataset import build_or_get_dataset, get_dataloader
from config import get_config
from wide_resnet import WideResNet

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using Device:", device)

trainset, testset_in, classes = build_or_get_dataset('cifar10', '../data')
_, testset_out, _= build_or_get_dataset('svhn', '../data')

trainloader = get_dataloader(trainset)
testloader_in = get_dataloader(testset_in)
testloader_out = get_dataloader(testset_out)
num_classes = len(classes)

config = get_config()
net = WideResNet(config['layers'], num_classes, config['widen_factor'], dropRate=config['droprate']).to(device)

cudnn.benchmark = True

optimizer = torch.optim.SGD(
    net.parameters(), config['learning_rate'], momentum=config['momentum'],
    weight_decay=config['decay'], nesterov=True)

def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lr_lambda=lambda step: cosine_annealing(
        step,
        config['epochs'] * len(trainloader),
        1,  # since lr_lambda computes multiplicative factor
        1e-6 / config['learning_rate']))



    
