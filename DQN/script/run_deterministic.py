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
# CAPA is Replay buffer capacity

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005
LR = 1e-4
CAPA = 10000 

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