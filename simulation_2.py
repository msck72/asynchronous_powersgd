import torch
import torch.nn.functional as F
from models import SimpleCNN, get_resnet18_for_cifar10
import torch.optim as optim
from dataset_preperation import MNIST_load_dataloaders, CIFAR10_load_dataloaders, CIFAR10_load_dataloaders_distributedly

import torch.distributed as dist

import random
from timer import Timer

import os

torch.multiprocessing.set_sharing_strategy('file_system')


def setup_groups(seed, group_cache):

    #Only wroks for two groups, shall be extended to n groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    torch.manual_seed(seed)
    permutation_of_nodes = torch.randperm(world_size).tolist()

    partitioner = int(world_size / 2)
    group1_indices = sorted(permutation_of_nodes[:partitioner])
    group2_indices = sorted(permutation_of_nodes[partitioner:])

    key1, key2 = '_'.join(map(str, group1_indices)), '_'.join(map(str, group2_indices))
    
    if key1 in group_cache:
        grp1, grp2 = group_cache[key1]
    else:
        grp1, grp2 = dist.new_group(group1_indices), dist.new_group(group2_indices)
        group_cache[key1] = (grp1, grp2)
        group_cache[key2] = (grp2, grp1)
    
    print(f"Groups: group1_indices  {group1_indices}, group2_indices = {group2_indices}\n\n")

    return grp1 if rank in group1_indices else grp2


def all_reduce_gradients(model, group):
    group_size = dist.get_world_size(group)

    with timer('all_reduce_gradients'):
        for name, param in model.named_parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
                # dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

                param.grad /= group_size

    print(f"{group_size}  all reduce graient done")


def synchronize_weights(model):
    world_size = dist.get_world_size()
    
    sync_state_dict = {}

    with timer('synchronize_weights'):
        for k, v in model.state_dict().items():
            if v.is_cuda:
                tensor_device = v.device
            else:
                tensor_device = torch.device('cpu')

            tensor_copy = v.clone()

            dist.all_reduce(tensor_copy, op=dist.ReduceOp.SUM)

            sync_state_dict[k] = tensor_copy / world_size

        model.load_state_dict(sync_state_dict)


def main():

    dist.init_process_group(backend='gloo')

    # Give DistributedDataSampler as input so that different nodes haev different examples
    # train_loader, test_loader = MNIST_load_dataloaders()
    # train_loader, test_loader = CIFAR10_load_dataloaders()

    train_loader, test_loader = CIFAR10_load_dataloaders_distributedly()

    model = get_resnet18_for_cifar10()

    train(model, train_loader)

    test(model, test_loader)


def train(model, train_loader):

    EPOCHS = 1
    SYNCHRONIZATION_FREQ = 20
    group_cache = {}

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = F.cross_entropy

    print(f"len of trainloder = {len(train_loader)}")

    model.train()
    for epoch in range(EPOCHS):
        print(f"epoch = {epoch}")
        for i, (images, targets) in enumerate(train_loader):
            optimizer.zero_grad()
            
            y_hat = model(images)

            loss = criterion(y_hat, targets)

            loss.backward()

            my_group = setup_groups(i, group_cache)
            
            print(f"New groups created\n\n")
            all_reduce_gradients(model, my_group)
            # all_reduce_gradients(model, None)


            my_group = None
            # print("Studk in all_reduce_gradients\n\n")

            optimizer.step()

            print(f"processing batch {i} done")
            if i % SYNCHRONIZATION_FREQ == 0:
                synchronize_weights(model)
                print(f"Synchronization done")
            
        synchronize_weights(model)
        print(f"Synchronization done")




def test(model, test_loader):

    test_loss = 0.0
    correct = 0
    total = 0

    model.eval()

    with torch.no_grad():
        for source, targets in test_loader:
        
            y_hat = model(source)
            loss = F.cross_entropy(y_hat, targets)

            test_loss += loss.item()

            # Compute accuracy (assuming classification)
            _, predicted = torch.max(y_hat, 1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    
    avg_loss = test_loss / len(test_loader)
    accuracy = correct / total * 100
    print(f"Test Loss: {avg_loss:.4f}, Test Accuracy: {accuracy:.2f}%")



def log_metric(name, values, tags={}):
    """Log timeseries data
    This function will be overwritten when called through run.py"""
    value_list = []
    for key in sorted(values.keys()):
        value = values[key]
        value_list.append(f"{value:7.3f}")
    values = ", ".join(value_list)
    tag_list = []
    for key, tag in tags.items():
        tag_list.append(f"{key}:{tag}")
    tags = ", ".join(tag_list)
    # print("{rank} {name:30s} - {values} ({tags})".format(rank=config['rank'], name=name, values=values, tags=tags))

    my_rank = dist.get_rank()
    os.makedirs(f"./logs/{my_rank}_logs", exist_ok=True)

    tags=tags.split(':')[1]
    with open(f"./logs/{my_rank}_logs/{tags}.txt", "a") as log_file:
        log_file.write("{rank},{tags:30s},{values}\n".format(rank=my_rank, name=name, values=values, tags=tags))

def metric(*args, **kwargs):
    # if config["rank"] == 0:
    #     log_metric(*args, **kwargs)
    
    log_metric(*args, **kwargs)



timer = Timer(log_fn=metric)

if __name__ == "__main__":
    main()



