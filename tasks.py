import torch
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from utils import setup_groups, all_reduce_gradients
from timer import Timer

from powersgd.powersgd import PowerSGD
from powersgd import optimizer_step, synchronize_weights, update_low_rank_weights

def train(model : torch.nn.Module, train_loader, test_loader, compressor : PowerSGD, timer: Timer, train_config):
    
    print(f"Trining the model with the following train_config: {train_config}")
    print(f"----------------HAPPY TRAINING---------------")
    

    

    EPOCHS = 5
    group_cache = {}
    # group_cache['default_group'] = dist.new_group([i for i in range(dist.get_world_size())])
    group_cache['default_group'] = dist.group.WORLD

    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = F.cross_entropy

    # test(model, test_loader, timer)



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

            my_group, my_group_id = setup_groups(i, group_cache, train_config['divide_groups'])

            if compressor:
                optimizer_step(optimizer, compressor, my_group, my_group_id, timer)
                if train_config['synchronize']:
                    update_low_rank_weights(optimizer, compressor, timer)
                
            else:
                all_reduce_gradients(model, my_group, timer)

            if train_config['synchronize'] and i != 0 and i % train_config['synchronization_freq'] == 0:
                synchronize_weights(optimizer, compressor, timer)

            my_group = None

            # is handled by the optimizer_step function
            # optimizer.step()
    
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