import torch
import torch.nn as nn
import numpy as np


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()
        self.mean = torch.tensor([0.4914, 0.4822, 0.4465]).view(1, 3, 1, 1)  # CIFAR-10 mean
        self.std = torch.tensor([0.2470, 0.2435, 0.2616]).view(1, 3, 1, 1)   # CIFAR-10 std

    def forward(self, x):
        return (x - self.mean.to(x.device)) / self.std.to(x.device)