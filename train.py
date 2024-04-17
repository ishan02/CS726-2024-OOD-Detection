import numpy as np
import os
import pickle
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from tqdm import tqdm
import logging

from build_dataset import build_or_get_dataset, get_dataloader
from config import get_config
from wide_resnet import WideResNet
config = get_config()
log_dir = './logs'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(config['folder'], exist_ok=True)

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.FileHandler(f"{log_dir}/training.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Log configuration settings
logger.info("Training configuration:")
for key, value in config.items():
    logger.info(f"{key}: {value}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using Device: {device}")

trainset, testset_in, classes = build_or_get_dataset('cifar10', '../data')
_, testset_out, _= build_or_get_dataset('svhn', '../data')

trainloader = get_dataloader(trainset, batch_size = config["batch_size"])
testloader_in = get_dataloader(testset_in, batch_size = config["batch_size"], shuffle= False)
testloader_out = get_dataloader(testset_out, batch_size = config["batch_size"], shuffle= False)
num_classes = len(classes)


net = WideResNet(config['layers'], num_classes, config['widen_factor'], dropRate=config['droprate']).to(device)

cudnn.benchmark = True

optimizer = torch.optim.SGD(
    net.parameters(), config['learning_rate'], momentum=config['momentum'],
    weight_decay=config['decay'], nesterov=True)
start_epoch = 0


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

if config['preload']:
    checkpoint_path = f"{config['save']}_{config['load_epoch']}.pth"
    checkpoint = torch.load(checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    scheduler.step(start_epoch * len(trainloader))  # Move scheduler to the appropriate step
    logger.info(f"Checkpoint loaded from {checkpoint_path}, at epoch {start_epoch}")

def train(net, device, dataloader, optimizer, criterion, epoch):
    net.train()
    running_loss = 0.0
    correct = 0
    total = 0
    progress_bar = tqdm(dataloader, desc='Training', leave=False)
    
    for batch_index, (inputs, targets) in enumerate(progress_bar):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar.set_postfix(loss=running_loss/(batch_index+1), accuracy=100.*correct/total)

    average_loss = running_loss / len(dataloader)
    accuracy = 100. * correct / total
    logger.info(f'Epoch {epoch+1}: Train Loss: {average_loss:.3f}, Train Accuracy: {accuracy:.2f}%')
    return average_loss, accuracy

def test(net, device, dataloader, epoch, output_type='both'):
    net.eval()
    correct = 0
    total = 0
    max_softmax_scores = []
    progress_bar = tqdm(dataloader, desc='Testing', leave=False)
    
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            softmax_scores = F.softmax(outputs, dim=1)
            max_scores = softmax_scores.max(dim=1).values
            max_softmax_scores.extend(max_scores.cpu().numpy())

            if output_type == 'both':
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

    if output_type == 'both':
        accuracy = 100. * correct / total
        logger.info(f'Epoch {epoch+1}: Test Accuracy: {accuracy:.2f}%')
        return {'accuracy': accuracy, 'max_softmax_scores': max_softmax_scores}

    return {'max_softmax_scores': max_softmax_scores}

criterion = nn.CrossEntropyLoss()

# Main training and testing loop
for epoch in range(start_epoch, config['epochs']):
    train_loss, train_acc = train(net, device, trainloader, optimizer, criterion, epoch)
    scheduler.step()

    state = {
        'epoch': epoch,
        'model_state_dict': net.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    path_save = f"{config['save']}_{epoch}.pth"
    torch.save(state, path_save)
    logger.info(f"Checkpoint saved at {path_save}")

    results_in = test(net, device, testloader_in, epoch, output_type='both')
    results_out = test(net, device, testloader_out, epoch, output_type='scores')


    logger.info(f'Epoch {epoch+1}: Max Softmax Scores (CIFAR-10 first 5): {results_in["max_softmax_scores"][:5]}')
    logger.info(f'Epoch {epoch+1}: Max Softmax Scores (SVHN first 5): {results_out["max_softmax_scores"][:5]}')
