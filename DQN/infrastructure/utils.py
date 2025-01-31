from infrastructure.config import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, CAPA, Transition, MAX_NORM
from infrastructure.pytorch_utils import device
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib


def optimize_model(memory, Q_net, target_net, optimizer):
    if len(memory) < BATCH_SIZE:
        return
    
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = Q_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
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
    