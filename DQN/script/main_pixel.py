import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.optim as optim
from itertools import count
from infrastructure.experience_replay import experience_replay
from model.DQN import QNetwork
from policy.epsilon_greedy_policy import epsilon_greedy_policy
from infrastructure.utils import optimize_model
from infrastructure.config import (
    GAMMA,
    EPS_START,
    EPS_END,
    EPS_DECAY,
    TAU,
    LR,
    BATCH_SIZE,
)
import gymnasium as gym

import numpy as np
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo, AtariPreprocessing
from gymnasium.wrappers import FrameStackObservation as FrameStack

# Function to run the experiment with a given seed
def run_experiment(seed):
    print(gym.envs.registry.keys())
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # Initialize environment, networks, optimizer, and replay buffer
    env = gym.make("BreakoutNoFrameskip-v4")  
    env = AtariPreprocessing(env)
    env = FrameStack(env, 4)  # Stack 4 frames

    n_act = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Q_net = QNetwork(n_act).to(device)
    target_net = QNetwork(n_act).to(device)
    target_net.load_state_dict(Q_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(Q_net.parameters(), lr=LR)
    memory = experience_replay()

    steps_done = 0
    epsilon_durations = []
    num_episodes = 5000 if torch.cuda.is_available() else 1000
    episode_rewards = []

    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

    env = RecordVideo(env, video_folder=f"video_seed_{seed}", episode_trigger=lambda episode_id: episode_id % 100 == 0)

    for i_episode in range(num_episodes):
        state = env.reset()[0]
        state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
        episode_reward = 0

        for t in count():
            action = epsilon_greedy_policy(state, Q_net, steps_done, EPS_START, EPS_END, EPS_DECAY, env)
            observations, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated

            next_state = None if terminated else torch.tensor(observations, device=device, dtype=torch.float).unsqueeze(0)

            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(memory, Q_net, target_net, optimizer, BATCH_SIZE, GAMMA)

            # Soft update the target network
            target_net_state_dict = target_net.state_dict()
            Q_net_state_dict = Q_net.state_dict()
            for key in Q_net_state_dict:
                target_net_state_dict[key] = (1 - TAU) * target_net_state_dict[key] + TAU * Q_net_state_dict[key]
            target_net.load_state_dict(target_net_state_dict)

            if done:
                episode_rewards.append(episode_reward)
                break

    # Save the plot of rewards
    plt.figure()
    plt.plot(episode_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Episode Rewards over Time (Seed {seed})')
    plt.savefig(f'episode_rewards_seed_{seed}.png')

    env.close()

# Run experiments with multiple seeds
seeds = [0, 1, 2]  # List of seeds to use
for seed in seeds:
    run_experiment(seed)