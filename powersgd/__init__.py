import torch
import torch.nn as nn

from powersgd.powersgd import Aggregator, AllReduce, Config, PowerSGD
from powersgd.utils import params_in_optimizer


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
        

def synchronize_weights(optimizer: torch.optim.Optimizer, aggregator, timer):
    optimizer.zero_grad()
    
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            params.append(p.clone().detach())

    
    sync_weights = aggregator.aggregate_parameters(params, timer)
    for p, sw in zip(params, sync_weights):
        p.copy_(sw)
    
    return

def update_low_rank_weights(optimizer : torch.optim.Optimizer, aggregator: Aggregator, timer) -> None:
    """
    Update the low rank matrices of weights as well to get the warm start type behaviour of synchronizing weights
    """
    
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            params.append(p.clone().detach())
      
    aggregator.update_low_rank_weights(params, timer)
    
    # if torch.distributed.get_rank() == 0:
    #     with torch.no_grad():
    #         tensor_lists = [p.numpy().tolist() for p in params_in_optimizer(optimizer)]
    #         with open(f"something_after_ulrw.txt", "w") as f:
    #             f.write(str(tensor_lists))
    
    return
