import torch
from torchvision import datasets
import torchvision.transforms as transforms

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

def get_loader(val_size=5000, batch_size=200):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    
    
    trainset, valset = torch.utils.data.random_split(trainset, [50000 - val_size, val_size],
                                                         torch.Generator().manual_seed(35))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, num_workers=0)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size, shuffle=False, num_workers=0)
    
    testset = datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size, shuffle=False, num_workers=0)
    
    return trainloader, valloader, testloader
