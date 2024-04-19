import torch
import torchvision
import os 
import sys 
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

# Get the parent directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
# Add the parent directory to the Python path
sys.path.append(parent_dir)

from build_dataset import build_or_get_dataset, get_dataloader, imshow

_, testset, classes = build_or_get_dataset('cifar10', '../data', True)
loader = get_dataloader(testset, batch_size = 30)

dataiter = iter(loader)
images, labels = next(dataiter)  # Get the first batch of images and labels

grid = make_grid(images[:30], nrow=6).permute(1,2,0)

plt.figure(figsize=(10,10))
plt.imshow(grid)
plt.savefig('./Glow/fig/dataset_images.png')
plt.show()
plt.axis('off')