import torch
import torch.nn.functional as F
from models import SimpleCNN, get_resnet18_for_cifar10
from dataset_preperation import MNIST_load_dataloaders, CIFAR10_load_dataloaders, CIFAR10_load_dataloaders_distributedly

import torch.distributed as dist

from timer import Timer

from tasks import train, test
from utils import metric

import yaml

from powersgd.powersgd_weights_sync import PowerSGD, Config

def main():

    with open('config.yaml', 'r') as file:
        setup_config = yaml.safe_load(file)

    dist.init_process_group(backend='gloo')

    timer = Timer(log_fn=metric)

    print("Hi")

    # Give DistributedDataSampler as input so that different nodes haev different examples
    # train_loader, test_loader = MNIST_load_dataloaders()
    # train_loader, test_loader = CIFAR10_load_dataloaders()

    
    train_loader, test_loader = CIFAR10_load_dataloaders_distributedly()

    model = get_resnet18_for_cifar10()

    compressor = None
    if setup_config['compress']:
        config = Config(rank=setup_config['compressor_conf']['rank'], 
                        min_compression_rate=setup_config['compressor_conf']['min_compression_rate'], 
                        num_iters_per_step=setup_config['compressor_conf']['num_iters_per_step'],
                        start_compressing_after_num_steps=setup_config['compressor_conf']['start_compressing_after_num_steps']
                    )

        compressor = PowerSGD(list(model.parameters()), config=config)
        print(f"Compressing the gradient using power-SGD compression mechanism, with the following config: {config}")
        print(f"----------------HAPPY COMPRESSION---------------")

    
    train(model, train_loader, test_loader, compressor=compressor, timer=timer, 
          train_config={'divide_groups': setup_config['divide_groups'], 
                        'num_groups': setup_config['num_groups'], 
                        'start_dividing_after': setup_config['start_dividing_after'],
                        'synchronize_weights': setup_config['synchronize_weights'], 
                        'synchronization_of_weights_freq': setup_config['synchronization_of_weights_freq'], 
                        'normal_synchronization': setup_config['normal_synchronization']})

    # test(model, test_loader, timer=timer)
    
    timer.save_summary("time_consumption_summary.json")
    timer.summary()


if __name__ == "__main__":
    main()



