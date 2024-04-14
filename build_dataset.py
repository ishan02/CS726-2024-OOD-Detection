import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import os

def build_or_get_dataset(name, root='../data'):
    assert os.path.exists(root), "data directory doesnot exist"
    if name == 'cifar10':
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,download=True, transform=transform_cifar10)
        testset = torchvision.datasets.CIFAR10(root=root, train=False,download=True, transform=transform_cifar10)
        classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    if name == 'mnist':
        trainset = torchvision.datasets.MNIST(root=root, train=True,download=True, transform=transform_mnist)
        testset = torchvision.datasets.MNIST(root=root, train=False,download=True, transform=transform_mnist)
        classes = tuple(str(i) for i in range(10))
    if name == 'svhn':
        trainset = torchvision.datasets.SVHN(root=root, split='train',download=True, transform=transform_svhn)
        testset = torchvision.datasets.SVHN(root=root, split='test',download=True, transform=transform_svhn)
        classes = tuple(str(i) for i in range(10))
    return trainset, testset, classes

def get_dataloader(set, batch_size = 64, shuffle= True):
    dataloader = torch.utils.data.DataLoader(set, batch_size=batch_size,shuffle=shuffle)
    return dataloader

transform_mnist = transforms.Compose([
    transforms.ToTensor(),  # Convert to tensor
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1] for grayscale images
])
transform_svhn = transforms.Compose([
    transforms.ToTensor(),                                   # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # Normalize pixel values to [-1, 1]
])
transform_cifar10 = transforms.Compose([
    transforms.ToTensor(),                                   # Convert image to PyTorch tensor
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   # Normalize pixel values to [-1, 1]
])

def imshow(img):
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    if npimg.shape[0] == 1:# Determine the number of color channels (1 for grayscale, 3 for RGB)
        plt.imshow(npimg[0], cmap='gray')# Grayscale image (1 channel): Convert to 2D array (H x W)
    else:# RGB image (3 channels): Transpose the array to (H x W x C) and display
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    print('in main')