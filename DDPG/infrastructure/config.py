import gymnasium as gym
from collections import namedtuple

# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor 
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
# CAPA is Replay buffer capacity
# MAX_NORM is the maximum norm of the gradients
# ACTSPACE is the action space of the environment
# NOISE_FACTOR is the factor of the noise added to the action

BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.5
EPS_END = 0.005
EPS_DECAY = 100
TAU = 0.005
LR_actor = 1e-4
LR_critic = 1e-4
CAPA = 10000
MAX_NORM = 50
NOISE_FACTOR = 1 
N_iter_GPU = 1000
N_iter_CPU = 100


Transition = namedtuple('Transition', ('state', 'action', 'reward','next_state'))
