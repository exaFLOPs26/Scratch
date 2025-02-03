import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import torch
import torch.optim as optim
from itertools import count
from infrastructure.replay_buffer import replay_buffer
from model.DDPG import QNetwork
from policy.action_net import actorNetwork, noise
from infrastructure.utils import optimize_model
from infrastructure.config import GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR_actor, LR_critic, BATCH_SIZE, NOISE_FACTOR
from infrastructure.pytorch_utils import init_gpu
import gymnasium as gym
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

# Initialize environment
try:
    env = gym.make("HalfCheetah-v5", render_mode="rgb_array")  
except Exception as e:
    print(f"Error initializing environment: {e}")
    sys.exit(1)

n_obs = env.observation_space.shape[0]
n_act = env.action_space.shape[0]

# Initialize critic and actor network
device = init_gpu()
actor_net = actorNetwork(n_obs, n_act).to(device)  # Actor to output continuous actions from observations
critic_net = QNetwork(n_obs, n_act).to(device)

# Initialize target network
target_actor_net = actorNetwork(n_obs, n_act).to(device)
target_critic_net = QNetwork(n_obs, n_act).to(device)
target_actor_net.load_state_dict(actor_net.state_dict())
target_critic_net.load_state_dict(critic_net.state_dict())
target_actor_net.eval()
target_critic_net.eval()

# Initialize optimizer
actor_optimizer = optim.Adam(actor_net.parameters(), lr=LR_actor)
critic_optimizer = optim.Adam(critic_net.parameters(), lr=LR_critic)

# Initialize replay buffer
memory = replay_buffer()

steps_done = 0
epsilon_durations = []
num_episodes = 5000 if torch.cuda.is_available() else 1000
episode_rewards = []

env = RecordVideo(env, video_folder="video", episode_trigger=lambda episode_id: episode_id % 100 == 0)

for i_episode in range(num_episodes):

    # Initialize a random process N for action exploration
    noise_action = noise(n_act, NOISE_FACTOR)
    
    # Receive initial observation state
    state, _ = env.reset()
    state = torch.tensor(state, device=device, dtype=torch.float).unsqueeze(0)

    episode_reward = 0
    
    for t in count():
        # Select action with added noise
        action = actor_net(state).detach().cpu() + noise_action
        
        # Execute action and observe reward and next state
        observations, reward, terminated, truncated, _ = env.step(np.squeeze(action.numpy()))
        next_state = None if terminated else torch.tensor(observations, device=device, dtype=torch.float).unsqueeze(0)
        reward = torch.tensor([reward], device=device)
        episode_reward += reward.item()
        
        # Store the transition in memory
        memory.push(state, action, reward, next_state)
        state = next_state

        optimize_model(memory, critic_net, target_critic_net, actor_net, target_actor_net, actor_optimizer, critic_optimizer, device)
        done = terminated or truncated
        
        if done:
            episode_rewards.append(episode_reward)
            break

# Save the plot of rewards
plt.figure()
plt.plot(episode_rewards)
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.title('Episode Rewards over Time')
plt.savefig('video/episode_rewards.png')

env.close()