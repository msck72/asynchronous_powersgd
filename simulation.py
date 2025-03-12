import torch
import torch.nn.functional as F
from models import SimpleCNN, get_resnet18_for_cifar10
import torch.optim as optim
from dataset_preperation import MNIST_load_dataloaders, CIFAR10_load_dataloaders

import random

EPOCHS = 1

def main():
    # train_loader, test_loader = MNIST_load_dataloaders()
    train_loader, test_loader = CIFAR10_load_dataloaders()

    my_model = train(train_loader)

    test(my_model, test_loader)

def train(train_loader):

    NODES = 10
    GROUPS = 2

    # Use this for MNISST
    # models = [SimpleCNN() for _ in range(NODES)]

    # Use thiss for CIFAR
    models = [get_resnet18_for_cifar10() for _ in range(NODES)]

    
    for i in range(1, len(models)):
        models[i].load_state_dict(models[0].state_dict())

    optimizers = [optim.SGD(models[i].parameters(), lr=0.001) for i in range(NODES)]

    
    aggregate_frequency = 50

    print("In train")
    for epoch in range(EPOCHS):
        # model.train()
        for i in range(NODES):
            models[i].train()
        
        for i, (source, targets) in enumerate(train_loader):
            # print(f"i == {i}")

            node = i % NODES

            # optimizers[node].zero_grad()

            y_hat = models[node](source)

            loss = F.cross_entropy(y_hat, targets)

            loss.backward()

            if node == NODES - 1:
                
                # group_grad_sum(models)

                # print(f"node = {node}")
                start = 0
                step = int(NODES / GROUPS)

                for _ in range(GROUPS):
                    group_grad_sum(models[start:(start + step)])
                    start = start + step

                for o in optimizers:
                    o.step()
                    o.zero_grad()

                random.shuffle(models)
            
                if (i + 1) % aggregate_frequency == 0:
                    print(f"i = {i}, synchronizing weights")
                    aggregate_weights(models)

                
    aggregate_weights(models)
    return models[0]


def group_grad_sum(models):
    # sum up the gradient 
    # print(len(models))
    effective_gradient = {}
    for params in zip(*[model.named_parameters() for model in models]):
        name = params[0][0]
        effective_gradient[name] = sum([param[1].grad for param in params])

    # for the first model optimizer.step and then copy the model parameters to all the other models...
    for m in models:
        for name, param in m.named_parameters():
            if param.grad is not None:
                param.grad = effective_gradient[name].clone()
    


def aggregate_weights(models):
    synchronized_weights = {}
    for k_v in zip(*[model.state_dict().items() for model in models]):
        k = k_v[0][0]
        synchronized_weights[k] = sum(v for _, v in k_v) / len(k_v)

    for model in models:
        model.load_state_dict({k: v.clone() for k, v in synchronized_weights.items()})



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


    
if __name__ == "__main__":
    main()


# NUM_OF_NODES = 10


# dist_models = [SimpleCNN() for _ in range(NUM_OF_NODES)]


# gradients = [None for _ in range(NUM_OF_NODES)]



# optimizers = []


