import torch
from torchvision import datasets
import torchvision.transforms as transforms
import random, numpy


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


g = torch.Generator()
g.manual_seed(0)

kwargs = {'num_workers': 1, 'pin_memory': True, 'worker_init_fn': seed_worker,
          'generator': g, }

def get_loader(val_size=5000, batch_size=128):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])

    trainset = datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    
    
    trainset, valset = torch.utils.data.random_split(trainset, [50000 - val_size, val_size],
                                                         torch.Generator().manual_seed(35))
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size, shuffle=True, **kwargs)

    valloader = torch.utils.data.DataLoader(
        valset, batch_size, shuffle=True, **kwargs)
    
    testset = datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    
    testloader = torch.utils.data.DataLoader(
        testset, batch_size, shuffle=True,**kwargs)
    
    return trainloader, valloader, testloader