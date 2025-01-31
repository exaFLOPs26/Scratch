import gymnasium as gym
from collections import namedtuple

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# CAPA is Replay buffer capacity
# MAX_NORM is the maximum norm of the gradients

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.2
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.005
LR = 1e-4
CAPA = 10000
MAX_NORM = 10

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
