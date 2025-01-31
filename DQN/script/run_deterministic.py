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

from DQN.network import DQN
from DQN.infrastructure.replay_buffer import ReplayBuffer
from DQN.infrastructure.pytorch_utils import device
from DQN.infrastructure.utils import optimize_model, plot_durations
from DQN.infrastructure.config import LR, CAPA, TAU
from DQN.policy.deterministic import d_policy


# Set up environment
env = gym.make('CartPole-v1')

# Get number of actions from gym action space
n_actions = env.action_space.n

# Get the number of state observations
state, info = env.reset()
n_observations = len(CAPA)

# Initialize the list of durations
epsilon_durations = []

Q_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(Q_net.state_dict())

optimizer = optim.AdamW(Q_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayBuffer(capacity= CAPA)

steps_done = 0

if torch.cuda.is_available():
    print("Using GPU")
    num_episodes = 600
else:
    print("Using CPU")
    num_episodes = 50
    
for i_episode in range(num_episodes):
    
    # Initialize the environment and state
    state, info = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
    for t in count():
        action = d_policy(state, Q_net, env)
        observations, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated
        
        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observations, device=device, dtype=torch.float).unsqueeze(0)
        
        # Store the transition in memory
        memory.push(state, action, next_state, reward)
        
        # Move to the next state
        state = next_state
        
        # Perform one step of the optimization
        optimize_model()
        
        # Soft update the target network using greedy policy
        
        target_net_state_dict = target_net.state_dict()
        Q_net_state_dict = Q_net.state_dict()
        
        for key in Q_net_state_dict:
            target_net_state_dict[key] = (1 - TAU) * target_net_state_dict[key] + TAU * Q_net_state_dict[key]
            
        target_net.load_state_dict(target_net_state_dict)
        
        if done:
            epsilon_durations.append(t + 1)
            plot_durations(epsilon_durations)
            break

print("Training complete")
plot_durations(show_result = True, epsilon_durations = epsilon_durations)
plt.ioff()
plt.show()
env.close()