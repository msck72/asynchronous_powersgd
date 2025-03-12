import torch
import torch.nn.functional as F
from models import SimpleCNN, get_resnet18_for_cifar10
from dataset_preperation import MNIST_load_dataloaders, CIFAR10_load_dataloaders, CIFAR10_load_dataloaders_distributedly

import torch.distributed as dist

from timer import Timer
import sys
import os


from tasks import train, test
from utils import metric

sys.path.append(os.path.abspath(os.path.join('..', 'powersgd', 'powersgd')))

from powersgd.powersgd import PowerSGD, Config

def main():

    dist.init_process_group(backend='gloo')

    timer = Timer(log_fn=metric)

    print("Hi")

    # Give DistributedDataSampler as input so that different nodes haev different examples
    # train_loader, test_loader = MNIST_load_dataloaders()
    # train_loader, test_loader = CIFAR10_load_dataloaders()

    
    train_loader, test_loader = CIFAR10_load_dataloaders_distributedly()

    model = get_resnet18_for_cifar10()

    config = Config(rank=1, start_compressing_after_num_steps=2)
    powersgd_compressor = PowerSGD(list(model.parameters()), config=config)

    train(model, train_loader, compressor=powersgd_compressor, timer=timer)

    test(model, test_loader, timer=timer)
    

if __name__ == "__main__":
    main()



