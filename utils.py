import torch
import torch.distributed as dist
import os


def setup_groups(seed, group_cache, divide_groups):

    if not divide_groups:
        return get_default_group(seed, group_cache)
    
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
    
    # print(f"Groups: group1_indices  {group1_indices}, group2_indices = {group2_indices}\n\n")
    # returning process group and group ID that I belong to
    return (grp1, group1_indices[0]) if rank in group1_indices else (grp2, group2_indices[0])

def get_default_group(seed, group_cache):

    return group_cache['default_group']


def all_reduce_gradients(model, group, timer):
    group_size = dist.get_world_size(group)

    with timer('all_reduce_gradients'):
        for name, param in model.named_parameters():
            if param.grad is not None:
                dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, group=group)
                # dist.all_reduce(param.grad, op=dist.ReduceOp.SUM)

                param.grad /= group_size

    # print(f"{group_size}  all reduce graient done")


def synchronize_weights(model, timer):
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
    
    my_rank = dist.get_rank()
    print("{rank} {name:30s} - {values} ({tags})".format(rank=my_rank, name=name, values=values, tags=tags))

    # os.makedirs(f"./logs/{my_rank}_logs", exist_ok=True)

    # tags=tags.split(':')[1]
    # with open(f"./logs/{my_rank}_logs/{tags}.txt", "a") as log_file:
    #     log_file.write("{rank},{tags:30s},{values}\n".format(rank=my_rank, name=name, values=values, tags=tags))

def metric(*args, **kwargs):
    # if config["rank"] == 0:
    #     log_metric(*args, **kwargs)
    
    log_metric(*args, **kwargs)