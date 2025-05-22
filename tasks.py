import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from utils import create_groups, all_reduce_gradients
from timer import Timer

from powersgd.powersgd import PowerSGD
from powersgd import optimizer_step, synchronize_weights, update_low_rank_weights, total_synchronize

def train(model : torch.nn.Module, train_loader, test_loader, compressor : PowerSGD, timer: Timer, train_config):
    
    print(f"Trining the model with the following train_config: {train_config}")
    print(f"----------------HAPPY TRAINING---------------")
    
    
    EPOCHS = 5
    group_cache = {}
    # group_cache['default_group'] = dist.new_group([i for i in range(dist.get_world_size())])
    group_cache['default_group'] = dist.group.WORLD

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = F.cross_entropy
    
    
    params_store = create_params_like_zeros(optimizer)
    print('Number of nodes * processes = ', dist.get_world_size())
    print(f'Size of the dataloader = {len(train_loader)}')
    
    accuracies = []

    for epoch in range(EPOCHS):
        print(f"epoch = {epoch}")
        model.train()
        for i, (images, targets) in enumerate(train_loader):
            # Is takn care by the optimizer_step function    
            # optimizer.zero_grad()
            
            y_hat = model(images)

            loss = criterion(y_hat, targets)

            with timer('backward'):
                loss.backward()

            start_dividing = False
            if train_config['divide_groups'] and (train_config['start_dividing_after'] < (epoch * len(train_loader) + i)):
                start_dividing = True
            my_group, my_group_id = create_groups(i, group_cache, start_dividing, train_config['num_groups'])
            
            # if torch.distributed.get_rank() == 0:
            #     print(f'i= {i} {start_dividing} {torch.distributed.get_world_size(my_group)}')
            # print(f'num_workers before = {torch.distributed.get_world_size(my_group)}')
            # print(my_group_id)

            if compressor:
                optimizer_step(optimizer, compressor, my_group, my_group_id, timer)
                # updation of low-rank estimates of the parameters for a warm start
                if i != 0 and train_config['divide_groups'] and train_config['synchronize_weights'] and not train_config['normal_synchronization']:
                    # if torch.distributed.get_rank() == 0:
                        # print('updated the lo-rank approximations of weights of the model')
                    update_low_rank_weights(optimizer, compressor, params_store, timer)
                
            else:
                all_reduce_gradients(model, my_group, timer)

            # if synchronization of weights with compressing them needs to be performed
            if train_config['synchronize_weights'] and not train_config['normal_synchronization'] and i != 0 and train_config['divide_groups'] and i % train_config['synchronization_of_weights_freq'] == 0 and (train_config['start_dividing_after'] < (epoch * len(train_loader) + i)):
                if torch.distributed.get_rank() == 0:
                    print('synchronoizing the approximated weights')
                synchronize_weights(optimizer, compressor, params_store, timer)
            
            # if normal-conventional synchronization of weights without compression needs to be performed
            if train_config['normal_synchronization'] and train_config['divide_groups'] and i != 0 and i % train_config['synchronization_of_weights_freq'] == 0 and (train_config['start_dividing_after'] < (epoch * len(train_loader) + i)):
                if torch.distributed.get_rank() == 0:
                    print(f'batch {i}, Synchronizing the actual weights')
                total_synchronize(optimizer, timer)

            my_group = None

            # is handled by the optimizer_step function
            # optimizer.step()
            if i != 0 and i % 100 == 0:
                print(f'batch num = {i}')
                accuracies.append((epoch * len(train_loader) + i, test(model, test_loader, timer)))
            
        accuracies.append(((epoch + 1) * len(train_loader), test(model, test_loader, timer)))
        
    print(accuracies)




def test(model, test_loader, timer):

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
    return accuracy
    

def create_params_like_zeros(optimizer):
    
    params = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            params.append(torch.zeros_like(p))
            
    return params