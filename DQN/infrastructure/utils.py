from DQN.infrastructure.config import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, CAPA, Transition, MAX_NORM
from DQN.infrastructure.pytorch_utils import device
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib


def optimize_model(memory, Q_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    transition = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transition))
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, 
                                            batch.next_state)), device = device, dtype = torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    Q_values = Q_net(state_batch).gather(1, action_batch)
    
    # Computing Value functions for all next state s_{t+1}
    next_state_values = torch.zeros(BATCH_SIZE, device = device)
    with torch.no_grad():
        next_state_values[non_final_mask] = Q_net(non_final_next_states).max(1).values
        
    # Compute the expected Q values
    expected_Q_values = (next_state_values * GAMMA) + reward_batch
    
    # Compute Huber loss
    cirterion = nn.SmoothL1Loss()
    loss = cirterion(Q_values, expected_Q_values.unsqueeze(1))
    
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    
    torch.nn.utils.clip_grad_norm_(Q_net.parameters(), MAX_NORM)
    optimizer.step()
    

def plot_durations(show_result = False, epsilon_durations=[]):
    # Set up matplotlib
    is_iptyhon = 'inline' in matplotlib.get_backend()
    if is_iptyhon:
        from IPython import display

    plt.ion()
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
    