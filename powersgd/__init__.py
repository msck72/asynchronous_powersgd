from typing import List
import torch
import torch.nn as nn

from powersgd.powersgd import Aggregator, AllReduce, Config, PowerSGD
from powersgd.utils import params_in_optimizer

import torch.distributed as dist


def optimizer_step(optimizer: torch.optim.Optimizer, aggregator: Aggregator, group, group_id, timer):
    """
    Aggregate gradients across workers using `aggregator`,
    and then take an optimizer step using the aggregated gradient.
    """
    params = params_in_optimizer(optimizer)
    grads = [p.grad.data for p in params]  # type: ignore
    avg_grads = aggregator.aggregate(grads, group, group_id, timer)  # subtracts the approximation from grads

    # Temporarily set parameter's gradients to the aggregated values
    for (p, g) in zip(params, avg_grads):
        p.grad = g

    # Run an optimizer step
    optimizer.step()

    # Put back the error buffer as the parameter's gradient
    for (p, g) in zip(params, grads):
        p.grad = g
        

def synchronize_weights(optimizer: torch.optim.Optimizer, aggregator: Aggregator, params_store: List[torch.Tensor], timer):
    optimizer.zero_grad()
    
    with torch.no_grad():
        i = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                params_store[i].copy_(p)
                i += 1

        
        sync_weights = aggregator.aggregate_parameters(params_store, timer)
        
        i = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                p.copy_(sync_weights[i])
                i += 1
    
    return

def update_low_rank_weights(optimizer : torch.optim.Optimizer, aggregator: Aggregator, params_store : List[torch.Tensor], timer) -> None:
    """
    Update the low rank matrices of weights as well to get the warm start type behaviour of synchronizing weights
    """

    with torch.no_grad():
        i = 0
        for group in optimizer.param_groups:
            for p in group["params"]:
                params_store[i].copy_(p)
                i += 1
        
        aggregator.update_low_rank_weights(params_store, timer)
    
    # if torch.distributed.get_rank() == 0:
    #     with torch.no_grad():
    #         tensor_lists = [p.numpy().tolist() for p in params_in_optimizer(optimizer)]
    #         with open(f"something_after_ulrw.txt", "w") as f:
    #             f.write(str(tensor_lists))
    
    return


def total_synchronize(optimizer : torch.optim.Optimizer, timer):
    world_size = dist.get_world_size()
    

    with timer('TOTAL_SYNCHRONIZING_WEIGHTS'):
        with torch.no_grad():
            i = 0
            for group in optimizer.param_groups:
                for p in group["params"]:
                    tensor_copy = p.clone().detach()
                    dist.all_reduce(tensor_copy, op=dist.ReduceOp.SUM)
                    tensor_copy.mul_(1 / world_size)
                    p.copy_(tensor_copy)
                    i += 1