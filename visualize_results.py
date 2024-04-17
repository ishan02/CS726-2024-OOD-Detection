import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from build_dataset import build_or_get_dataset, get_dataloader
from config import get_config
from wide_resnet import WideResNet

config = get_config()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

_, testset_in, classes = build_or_get_dataset('cifar10', '../data')
_, testset_out, _= build_or_get_dataset('svhn', '../data')
testloader_in = get_dataloader(testset_in, batch_size = config["batch_size"], shuffle= False)
testloader_out = get_dataloader(testset_out, batch_size = config["batch_size"], shuffle= False)
num_classes = len(classes)

net = WideResNet(config['layers'], num_classes, config['widen_factor'], dropRate=config['droprate']).to(device)
criterion = nn.CrossEntropyLoss()

checkpoint_path = f"{config['save']}_{config['load_epoch']}.pth"
checkpoint = torch.load(checkpoint_path)
net.load_state_dict(checkpoint['model_state_dict'])
print(f"Checkpoint loaded from {checkpoint_path}")

def test(net, device, dataloader):
    net.eval()
    max_softmax_scores = []
    progress_bar = tqdm(dataloader, desc='Testing', leave=False)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            softmax_scores = F.softmax(outputs, dim=1)
            max_scores = softmax_scores.max(dim=1).values
            max_softmax_scores.extend(max_scores.cpu().numpy())

    return {'max_softmax_scores': max_softmax_scores}


def plot_softmax_histogram(max_softmax_scores_in, max_softmax_scores_out):
    plt.figure(figsize=(10, 6))
    sns.histplot(max_softmax_scores_in, bins=500, kde=True, label='CIFAR-10', color='blue', alpha=0.7, stat='density')
    sns.histplot(max_softmax_scores_out, bins=500, kde=True, label='SVHN', color='red', alpha=0.7, stat='density')
    plt.xlabel('Maximum Softmax Score')
    plt.ylabel('Density')
    plt.title('Maximum Softmax Score Distribution: CIFAR-10 vs. SVHN')
    plt.legend()
    plt.savefig('softmax_score_histogram.png')
    plt.show()


max_softmax_scores_in = test(net, device, testloader_in)
max_softmax_scores_out = test(net, device, testloader_out)

plot_softmax_histogram(max_softmax_scores_in, max_softmax_scores_out)