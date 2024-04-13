import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


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
def load_cifar10(batch_size=64, shuffle=True):

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform_cifar10)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=shuffle, num_workers=2)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform_cifar10)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    return trainloader, testloader, classes


def load_mnist(batch_size=64, shuffle=True):
    trainset = torchvision.datasets.MNIST(root='./data', train=True,download=True, transform=transform_mnist)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=shuffle, num_workers=2)
    testset = torchvision.datasets.MNIST(root='./data', train=False,download=True, transform=transform_mnist)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    classes = tuple(str(i) for i in range(10))
    return trainloader, testloader, classes

def load_svhn(batch_size=64, shuffle=True):
    trainset = torchvision.datasets.SVHN(root='./data', split='train',download=True, transform=transform_svhn)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=shuffle, num_workers=2)
    testset = torchvision.datasets.SVHN(root='./data', split='test',download=True, transform=transform_svhn)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
    classes = tuple(str(i) for i in range(10))
    return trainloader, testloader, classes


def imshow(img):
    # Unnormalize and convert tensor to numpy array
    img = img / 2 + 0.5  # Unnormalize
    npimg = img.numpy()
    # Determine the number of color channels (1 for grayscale, 3 for RGB)
    if npimg.shape[0] == 1:
        # Grayscale image (1 channel): Convert to 2D array (H x W)
        plt.imshow(npimg[0], cmap='gray')
    else:
        # RGB image (3 channels): Transpose the array to (H x W x C) and display
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


if __name__ == "__main__":
    
    
    # Load MNIST
    mnist_trainloader, mnist_testloader, mnist_classes = load_mnist()
    print("MNIST classes:", mnist_classes)
    dataiter = iter(mnist_trainloader)
    images, labels = next(dataiter)  # Get the first batch of images and labels
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % mnist_classes[labels[j]] for j in range(len(images))))
    
    '''
    cifar_trainloader, cifar_testloader, cifar_classes = load_cifar10()
    dataiter = iter(cifar_trainloader)
    images, labels = next(dataiter)  
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % cifar_classes[labels[j]] for j in range(len(images))))
    '''
    ''''
    svhn_trainloader, svhn_testloader, svhn_classes = load_svhn()
    dataiter = iter(svhn_trainloader)
    images, labels = next(dataiter)  
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % svhn_classes[labels[j]] for j in range(len(images))))
    '''