import gymnasium as gym
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
from infrastructure.config import Batch_size, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, CAPA

env = gym.make('CartPole-v1')

# set up matplotlib
is_iptyhon = 'inline' in matplotlib.get_backend()
if is_iptyhon:
    from IPython import display

plt.ion()

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(CAPA)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(capacity= CAPA)


steps_done = 0

# As mentioned by the name of this file, we will be implementing a deterministic policy.
def select_action(state): 
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    
    if sample > eps_threshold: # Select the action that has the maximum expected return
        with torch.no_grad():
            return policy_net(state).max(1)[1].view(1, 1)
    else: # Select a random action - exploration (epsilon-greedy)
        return torch.tensor([[random.randrange(n_actions)], device=device, dtype=torch.long])
    
epsilon_durations = []

def plot_durations(show_result = False):
    plt.figure(1)
    duration_t = torch.tensor(epsilon_durations, dtype=torch.float)
    
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.xlabel("Episode")
    plt.ylabel("Duration")
    plt.plot(duration_t.numpy())
    
    if len(duration_t) >= 100:
        means = duration_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())
        
    plt.pause(0.001) # pause a bit so that plots are updated
    
    if is_iptyhon:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())