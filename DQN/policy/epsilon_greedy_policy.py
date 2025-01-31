import torch
import random
import math
from infrastructure.config import EPS_DECAY, EPS_END, EPS_START

def epsilon_greedy_policy(state, Q_net, steps_done, env):
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return Q_net(state).max(1)[1].view(1, 1)
    else:
        return torch.tensor([[random.randrange(env.action_space.n)]], device=state.device, dtype=torch.long)
   