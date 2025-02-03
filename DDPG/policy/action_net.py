import torch
import torch.nn as nn
import torch.nn.functional as F

def noise(n_act, noise_factor):
    return torch.randn(n_act) * noise_factor

class actorNetwork(nn.Module):
    def __init__(self, state_size, action_size):
        super(actorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.bn1 = nn.BatchNorm1d(400)
        self.fc2 = nn.Linear(400, 300)
        self.bn2 = nn.BatchNorm1d(300)
        self.fc3 = nn.Linear(300, action_size)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        x = torch.tanh(self.fc3(x))  # Use tanh to ensure actions are within the range [-1, 1]
        return x