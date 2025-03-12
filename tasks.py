import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from utils import setup_groups, all_reduce_gradients, synchronize_weights
from timer import Timer

from powersgd.powersgd import PowerSGD

def train(model : torch.nn.Module, train_loader, test_loader, compressor : PowerSGD, timer: Timer, train_config):
    
    print(f"Trining the model with the following train_config: {train_config}")
    print(f"----------------HAPPY TRAINING---------------")
    

    EPOCHS = 5
    group_cache = {}
    # group_cache['default_group'] = dist.new_group([i for i in range(dist.get_world_size())])
    group_cache['default_group'] = dist.group.WORLD

    optimizer = optim.SGD(model.parameters(), lr=0.001)
    criterion = F.cross_entropy

    print(f"len of trainloder = {len(train_loader)}")

    for epoch in range(EPOCHS):
        print(f"epoch = {epoch}")
        model.train()
        for i, (images, targets) in enumerate(train_loader):    
            optimizer.zero_grad()
            
            y_hat = model(images)

            loss = criterion(y_hat, targets)

            with timer('backward'):
                loss.backward()

            my_group = setup_groups(i, group_cache, train_config['divide_groups'])
            
            # print(f'group size = {dist.get_world_size(my_group)}')
            # print(f"New groups created\n\n")

            # reduce the gradients and find the effective gradient
            # To do so update the powersgd by sending the group as well...

            if compressor:
                gradients = [param.grad for param in model.parameters()]

                with timer("aggregate_graddients", i):
                    aggregated_gradients = compressor.aggregate(gradients = gradients, dist_group=my_group, timer=timer)

                with timer('copying_agg_gradients', i):
                    for param, agg_grad in zip(model.parameters(), aggregated_gradients):
                        if param.grad is not None:
                            param.grad.copy_(agg_grad)
            else:
                all_reduce_gradients(model, my_group, timer)


            my_group = None
            # print("Stuck in all_reduce_gradients\n\n")

            optimizer.step()

            # print(f"processing batch {i} done")
        #     if i != 0 and train_config['divide_groups'] and i % int(train_config['synchronization_freq']) == 0:
        #         with torch.no_grad():
        #             if compressor:
        #                 aggregated_weights = compressor.aggregate_parameters(list(model.parameters()), dist_group=group_cache['default_group'], timer=timer)
                        
        #                 for param, new_val in zip(model.parameters(), aggregated_weights):
        #                     param.data.copy_(new_val) 
        #             else:
        #                 synchronize_weights(model, timer=timer)
        #         print(f"Synchronization done")
            
        # if train_config['divide_groups']:
        #     with torch.no_grad():
        #         if compressor:
        #                 aggregated_weights = compressor.aggregate_parameters(list(model.parameters()), dist_group=group_cache['default_group'], timer=timer)

        #                 for param, new_val in zip(model.parameters(), aggregated_weights):
        #                     param.data.copy_(new_val)
        #         else:
        #             synchronize_weights(model, timer=timer)
        #     print(f"Synchronization done")
    
        test(model, test_loader, timer)
        




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