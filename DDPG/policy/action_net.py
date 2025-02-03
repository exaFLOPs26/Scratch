import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def noise(n_act, noise_factor):
    return torch.randn(n_act) * noise_factor

class actorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(actorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x