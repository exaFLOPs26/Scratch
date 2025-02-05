from infrastructure.config import BATCH_SIZE, GAMMA, TAU, Transition
import torch
import torch.nn.functional as F


def optimize_model(
    memory,
    critic_net,
    target_critic_net,
    actor_net,
    target_actor_net,
    actor_optimizer,
    critic_optimizer,
    device,
):
    # Sample a random minibatch of transitions from the replay buffer
    if len(memory) < BATCH_SIZE:
        return

    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Update critic network by minimizing the loss
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device,
        dtype=torch.bool,
    )
    non_final_next_states = torch.cat(
        [s for s in batch.next_state if s is not None]
    ).to(device, dtype=torch.float32)
    
    state_batch = torch.stack(batch.state).to(device, dtype=torch.float32)
    # action_batch = torch.cat(
    #     [torch.tensor(a, device=device, dtype=torch.float32) for a in batch.action]
    # ).to(device, dtype=torch.float32)
    action_batch = torch.stack(batch.action).to(device, dtype=torch.float32)
    reward_batch = torch.cat(batch.reward).to(device, dtype=torch.float32)

    # Compute the target Q value
    next_actions = target_actor_net(non_final_next_states)
    next_state_values = torch.zeros(BATCH_SIZE, device=device, dtype=torch.float32)
    next_state_values[non_final_mask] = target_critic_net(
        non_final_next_states, next_actions
    ).squeeze()
    expected_state_action_values = (reward_batch + GAMMA * next_state_values).detach()

    # Compute the current Q value
    state_action_values = critic_net(state_batch, action_batch).squeeze()

    # Compute critic loss
    critic_loss = F.mse_loss(state_action_values, expected_state_action_values)

    # Optimize the critic network
    critic_optimizer.zero_grad()
    critic_loss.backward()
    critic_optimizer.step()

    # Update actor network by maximizing the expected return
    actor_optimizer.zero_grad()
    policy_loss = -critic_net(state_batch, actor_net(state_batch)).mean()
    policy_loss.backward()
    actor_optimizer.step()

    # Soft update target networks
    with torch.no_grad():
        for target_param, param in zip(target_actor_net.parameters(), actor_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
        for target_param, param in zip(target_critic_net.parameters(), critic_net.parameters()):
            target_param.data.copy_(TAU * param.data + (1.0 - TAU) * target_param.data)
