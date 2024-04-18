import torch
import torchvision
import os 
import sys 

# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from build_dataset import build_or_get_dataset, get_dataloader, imshow

trainset, testset, classes = build_or_get_dataset('svhn', '../data')
trainloader = get_dataloader(trainset)

dataiter = iter(trainloader)
images, labels = next(dataiter)  # Get the first batch of images and labels
print(' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
imshow(torchvision.utils.make_grid(images))

