import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import math
import random
import matplotlib.pyplot as plt
import matplotlib

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQN.infrastructure.pytorch_utils import device
from infrastructure.config import EPS_START, EPS_END, EPS_DECAY

steps_done = 0

# As mentioned by the name of this file, we will be implementing a deterministic policy.
def d_policy(state, Q_net, env): 
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    # Select the action that has the maximum expected return
    if sample > eps_threshold: 
        with torch.no_grad():
            return Q_net(state).max(1)[1].view(1, 1)
    
    # Select a random action ~ exploration (epsilon-greedy)
    else: 
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
   