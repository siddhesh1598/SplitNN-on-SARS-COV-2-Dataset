# import
import torch

from torch.nn import Sequential, Linear, ReLU, LogSoftmax
from torch.optim import SGD


# SplitNN model
class SplitNN:
    
    # initializing models and optimizers
    def __init__(self, models, optimizers):
        self.models = models
        self.optimizers = optimizers
        
    # forward pass to the model
    def forward(self, x):
        a = []
        remote_a = []
        
        a.append(self.models[0](x))
        if a[-1].location == self.models[1].location:
            remote_a.append(a[-1].detach().requires_grad_())
        else:
            remote_a.append(
                a[-1].detach().move(self.models[1].location).requires_grad_()
            )

        i = 1    
        while i < (len(self.models)-1):
            
            a.append(self.models[i](remote_a[-1]))
            if a[-1].location == self.models[i+1].location:
                remote_a.append(a[-1].detach().requires_grad_())
            else:
                remote_a.append(
                    a[-1].detach().move(self.models[i+1].location).requires_grad_()
                )
            
            i += 1
        
        a.append(self.models[i](remote_a[-1]))
        self.a = a
        self.remote_a = remote_a
        
        return a[-1]
    
    # backward pass / back propagation
    def backward(self):
        a = self.a
        remote_a = self.remote_a
        optimizers = self.optimizers
        
        i = len(self.models) - 2   
        while i > -1:
            if remote_a[i].location == a[i].location:
                grad_a = remote_a[i].grad.copy()
            else:
                grad_a = remote_a[i].grad.copy().move(a[i].location)
            a[i].backward(grad_a)
            i -= 1

    # initializing the gradients to zero
    def zero_grads(self):
        for opt in self.optimizers:
            opt.zero_grad()
    
    # optimizing the gradients
    def step(self):
        for opt in self.optimizers:
            opt.step()


# split models for different users
def splitModels(args, hidden_sizes=[128, 640]):
    models = [
        Sequential(
                    Linear(args.input_size, hidden_sizes[0]),
                    ReLU(),
        ),
        Sequential(
                    Linear(hidden_sizes[0], hidden_sizes[1]),
                    ReLU(),
        ),
        Sequential(
                    Linear(hidden_sizes[1], args.output_size),
                    LogSoftmax(dim=1)
        )
    ]

    return models


# split optimizers for each model
def splitOptimizers(models, args):
    optimizers = [
        SGD(model.parameters(), lr=args.lr)
        for model in models
    ]

    return optimizers