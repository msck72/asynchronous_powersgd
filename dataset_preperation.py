import torch

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler



def MNIST_load_dataloaders():

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    trainset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    testset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader


def CIFAR10_load_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32, shuffle=False)

    return trainloader, testloader



def CIFAR10_load_dataloaders_distributedly():

    BATCH_SIZE = 32

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))
    ])

    trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    sampler = DistributedSampler(trainset)

    trainloader = DataLoader(
            trainset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
        )

    testloader = DataLoader(
            testset,
            batch_size=BATCH_SIZE,
            pin_memory=True,
            drop_last=True,
            num_workers=2,
        )

    return trainloader, testloader




