import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQN.network.DQN import DQN
from DQN.infrastructure.replay_buffer import ReplayBuffer
from DQN.infrastructure.pytorch_utils import device

env = gym.make('CartPole-v1')

# set up matplotlib
is_iptyhon = 'inline' in matplotlib.get_backend()
if is_iptyhon:
    from IPython import display

plt.ion()



# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(10000)


steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    