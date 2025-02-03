# Replay buffer (which is called Experience Replay in the DQN paper) is a memory that stores transitions that the agent observes, allowing us to reuse this data later.
# By sampling from it randomly, the transitions that build up a batch are decorrelated. 
# It has been shown that this greatly stabilizes and improves the DQN training procedure.
from collections import deque
import random
from infrastructure.config import Transition, CAPA

class replay_buffer(object):
    def __init__(self):
        self.memory = deque([], maxlen = CAPA)
        
    def push(self, *args):
        # Save a transition
        self.memory.append(Transition(*args)) # replay_buffer.push(state, action, next_state, reward)
        
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory) # In the case of __len__, it is a special method that allows an object to implement behavior for the built-in len() function. When you call len() on an instance of a class that has a __len__ method, Python will call that method to get the length of the object.
    