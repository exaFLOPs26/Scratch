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

import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

# Initialize environment, networks, optimizer, and replay buffer
env = gym.make(
    "MountainCar-v0", render_mode="rgb_array"
)  # Initialize your environment here
n_obs = env.observation_space.shape[0]
n_act = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
Q_net = QNetwork(n_obs, n_act).to(device)
target_net = QNetwork(n_obs, n_act).to(device)
target_net.load_state_dict(Q_net.state_dict())
target_net.eval()
optimizer = optim.Adam(Q_net.parameters(), lr=LR)
memory = experience_replay()

steps_done = 0
epsilon_durations = []
num_episodes = 5000 if torch.cuda.is_available() else 1000
episode_rewards = []

print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

env = RecordVideo(
    env, video_folder="video", episode_trigger=lambda episode_id: episode_id % 100 == 0
)

for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
    episode_reward = 0

    for t in count():
        action = epsilon_greedy_policy(state, Q_net, steps_done, env)

        observations, reward, terminated, truncated, _ = env.step(action.item())

        next_state = (
            None
            if terminated
            else torch.tensor(observations, device=device, dtype=torch.float).unsqueeze(
                0
            )
        )

        reward = torch.tensor([reward], device=device)
        episode_reward += reward.item()

        memory.push(state, action, next_state, reward)
        state = next_state
        optimize_model(memory, Q_net, target_net, optimizer, device)

        # Soft update the target network (Idea from DDPG)
        target_net_state_dict = target_net.state_dict()
        Q_net_state_dict = Q_net.state_dict()
        for key in Q_net_state_dict:
            target_net_state_dict[key] = (1 - TAU) * target_net_state_dict[
                key
            ] + TAU * Q_net_state_dict[key]
        target_net.load_state_dict(target_net_state_dict)

        done = terminated or truncated

        if done:
            episode_rewards.append(episode_reward)
            break

# Save the plot of rewards
plt.figure()
plt.plot(episode_rewards)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Rewards over Time")
plt.savefig("video/episode_rewards.png")

env.close()
