#contributed #biulds or load the dataset
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os
import torch.nn.functional as F

def build_or_get_dataset(name, root='../data',task_generation=False):
    assert os.path.exists(root), "data directory doesnot exist"
    if task_generation:
        transform = transforms.Compose([transforms.ToTensor(), preprocess])
        target_transform = one_hot_encode
    else :
        transform = transform_rgb
        target_transform = None
    if name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,download=True, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False,download=True, transform=transform, target_transform=target_transform)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=root, train=True,download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root=root, train=False,download=True, transform=transform_mnist)
        classes = tuple(str(i) for i in range(10))
    if name == 'svhn':
        trainset = torchvision.datasets.SVHN(root=root, split='train',download=True, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.SVHN(root=root, split='test',download=True, transform=transform, target_transform=target_transform)
        classes = tuple(str(i) for i in range(10))
    return trainset, testset, classes

def get_dataloader(set, batch_size = 64, shuffle= True, drop_last=False):
    dataloader = torch.utils.data.DataLoader(set, batch_size=batch_size,shuffle=shuffle,drop_last=drop_last)
    return dataloader

transform_mnist = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for grayscale images
])
transform_rgb = transforms.Compose([
    transforms.ToTensor(),                                   # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # Normalize pixel values to [-1, 1]
])



n_bits = 8
def preprocess(x):#quantization
    x = x * 255  # undo ToTensor scaling to [0,1]
    n_bins = 2 ** n_bits
    if n_bits < 8:
        x = torch.floor(x / 2 ** (8 - n_bits))
    x = x / n_bins - 0.5
    return x
def postprocess(x): #dequantization
    x = torch.clamp(x, -0.5, 0.5)
    x += 0.5
    x = x * 2 ** n_bits
    return torch.clamp(x, 0, 255).byte()
def one_hot_encode(target):
    num_classes = 10
    one_hot_encoding = F.one_hot(torch.tensor(target),num_classes)
    return one_hot_encoding





def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if npimg.shape[0] == 1:# Determine the number of color channels (1 for grayscale, 3 for RGB)
        plt.imshow(npimg[0], cmap='gray')# Grayscale image (1 channel): Convert to 2D array (H x W)
    else:# RGB image (3 channels): Transpose the array to (H x W x C) and display
        plt.figure(figsize=(10,10))
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        
    plt.show()
    plt.axis('off')


if __name__ == "__main__":
    print('in main')