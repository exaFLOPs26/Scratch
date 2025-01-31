# Setting

## CartPole

As the agent observes the current state of the environment and chooses an action, the environment transitions to a new state, and also returns a reward that indicates the consequences of the action. In this task, **rewards are +1 for every incremental timestep** and the environment **terminates if the pole falls over too far or the cart moves more than 2.4 units away from center**. This means better performing scenarios will run for longer duration, accumulating larger return.
The CartPole task is designed so that the inputs to the agent are 4 real values representing the environment state (position, velocity, etc.). We take these 4 inputs without any scaling and pass them through a small fully-connected network with 2 outputs, one for each action. The network is trained to predict the expected value for each action, given the input state. The action with the highest expected value is then chosen.

# Run
cd DQN

`python3 script/main.py`

`python video/video_list.py `

`ffmpeg -f concat -safe 0 -i DQN/video/video_list.txt -c copy DQN/video/video.mp4`

# Algorithm
![image info](./image/algo_DQN.png)

- Initialize 
- With probability epsilon select a random action a_t
    - script/run_deterministic.py line: 75
- otherwise select a_t = max_{a} Q*(pi(s_t),a ;  theta)
    - script/run_deterministic.py line: 72