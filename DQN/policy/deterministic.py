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

from DQN.network.DQN import DQN
from DQN.infrastructure.replay_buffer import ReplayBuffer
from DQN.infrastructure.pytorch_utils import device
from infrastructure.config import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, CAPA, env, n_actions, n_observations

steps_done = 0

# As mentioned by the name of this file, we will be implementing a deterministic policy.
def select_action(state, policy_net): 
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold: # Select the action that has the maximum expected return
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else: # Select a random action - exploration (epsilon-greedy)
        return torch.tensor([[random.randrange(n_actions)], device=device, dtype=torch.long])
   