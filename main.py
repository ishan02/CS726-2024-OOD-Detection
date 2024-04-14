import torch
import torchvision


from build_dataset import build_or_get_dataset, get_dataloader, imshow

trainset, testset, classes = build_or_get_dataset('svhn', '../data')
trainloader = get_dataloader(trainset)

dataiter = iter(trainloader)
images, labels = next(dataiter)  # Get the first batch of images and labels
print(' '.join('%5s' % classes[labels[j]] for j in range(len(images))))
imshow(torchvision.utils.make_grid(images))

