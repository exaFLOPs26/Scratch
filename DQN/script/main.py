import gymnasium as gym
from gymnasium.wrappers import RecordVideo

import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from itertools import count

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from network.DQN import DQN
from infrastructure.replay_buffer import ReplayBuffer
from infrastructure.pytorch_utils import device
from infrastructure.utils import optimize_model, plot_durations
from infrastructure.config import LR, CAPA, TAU
from policy.deterministic import d_policy

def train_dqn():
    
    # Set up environment
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    env = RecordVideo(env, video_folder="DQN/video", episode_trigger=lambda episode_id: True)

    # Info about the environment
    n_actions = env.action_space.n
    state, info = env.reset()
    n_observations = len(state)

    # Initialize networks, optimizer, and replay buffer
    Q_net = DQN(n_observations, n_actions).to(device)
    target_net = DQN(n_observations, n_actions).to(device)
    
    target_net.load_state_dict(Q_net.state_dict())
    
    optimizer = optim.AdamW(Q_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayBuffer(capacity=CAPA)
    
    steps_done = 0
    epsilon_durations = []
    num_episodes = 600 if torch.cuda.is_available() else 50
    
    print(f"Using {'GPU' if torch.cuda.is_available() else 'CPU'}")

    for i_episode in range(num_episodes):
        state, info = env.reset()
        state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)
        
        for t in count():
            action = d_policy(state, Q_net, env)
            observations, reward, terminated, truncated, _ = env.step(action.item())
            reward = torch.tensor([reward], device=device)
            done = terminated or truncated
            
            next_state = None if terminated else torch.tensor(observations, device=device, dtype=torch.float).unsqueeze(0)
            
            memory.push(state, action, next_state, reward)
            state = next_state
            optimize_model(memory, Q_net, optimizer)
            
            # Soft update the target network
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
    plot_durations(show_result=True, epsilon_durations=epsilon_durations)
    plt.ioff()
    plt.show()
    env.close()

if __name__ == "__main__":
    train_dqn()
