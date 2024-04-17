import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from build_dataset import build_or_get_dataset, get_dataloader
from config import get_config
from wide_resnet import WideResNet
import logging

config = get_config()
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"{log_dir}/results.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info("Training configuration:")
for key, value in config.items():
    logger.info(f"{key}: {value}")

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

    return max_softmax_scores


def plot_softmax_histogram(max_softmax_scores_in, max_softmax_scores_out):
    plt.figure(figsize=(10, 6))
    sns.histplot(max_softmax_scores_in, bins=500, kde=True, label='CIFAR-10', color='blue', alpha=0.7, stat='density')
    sns.histplot(max_softmax_scores_out, bins=500, kde=True, label='SVHN', color='red', alpha=0.7, stat='density')
    plt.xlabel('Maximum Softmax Score')
    plt.ylabel('Density')
    plt.title('Maximum Softmax Score Distribution: CIFAR-10 vs. SVHN')
    plt.legend()
    plt.savefig('./fig/softmax_score_histogram.png')
    plt.show()


max_softmax_scores_in = test(net, device, testloader_in)
max_softmax_scores_out = test(net, device, testloader_out)
print("max_softmax_scores_in shape:", np.shape(max_softmax_scores_in))
print("max_softmax_scores_out shape:", np.shape(max_softmax_scores_out))

plot_softmax_histogram(max_softmax_scores_in, max_softmax_scores_out)

def calculate_fpr95(max_softmax_scores_in, max_softmax_scores_out):
    # Determine the TNR (True Negative Rate) threshold for 95% TNR
    num_in = len(max_softmax_scores_in)
    num_out = len(max_softmax_scores_out)
    total_negatives = num_in + num_out
    
    # Sort maximum softmax scores (higher score indicates more confident ID prediction)
    all_scores = np.concatenate([max_softmax_scores_in, max_softmax_scores_out])
    labels = np.concatenate([np.zeros(num_in), np.ones(num_out)])  # 0 for in-distribution, 1 for out-of-distribution
    
    # Determine threshold for achieving 95% TNR
    thresholds = np.sort(all_scores)
    tnr_threshold_index = int(0.95 * num_in)  # Index for 95% TNR
    tnr_threshold = thresholds[tnr_threshold_index]

    # Calculate FPR at 95% TNR
    fpr95 = np.sum((all_scores >= tnr_threshold) & (labels == 1)) / num_out
    
    return fpr95

def calculate_auroc(max_softmax_scores_in, max_softmax_scores_out):
    # Concatenate scores and labels
    all_scores = np.concatenate([max_softmax_scores_in, max_softmax_scores_out])
    labels = np.concatenate([np.zeros(len(max_softmax_scores_in)), np.ones(len(max_softmax_scores_out))])
    
    # Compute ROC curve and AUROC
    fpr, tpr, _ = roc_curve(labels, all_scores, pos_label=1)
    auroc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUROC = {auroc:.2f}')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.savefig('./fig/ROC Curve.png')
    plt.legend(loc='lower right')
    plt.show()

    return auroc



fpr95 = calculate_fpr95(max_softmax_scores_in, max_softmax_scores_out)
auroc= calculate_auroc(max_softmax_scores_in, max_softmax_scores_out)
print(f"AUROC: {auroc:.4f}")

logger.info(f"AUROC: {auroc:.4f} FPR95: {fpr95:.4f}")