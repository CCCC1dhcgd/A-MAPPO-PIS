
import random
from collections import namedtuple


Experience = namedtuple("Experience",
                        ("states", "actions", "rewards", "policies", "action_masks", "next_states", "dones"))


class ReplayMemory(object):
    """
    Replay memory buffer
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def _push_one(self, state, action, reward, policies, action_masks, next_state=None, done=None):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Experience(state, action, reward, policies, action_masks, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def push(self, states, actions, rewards, policies, action_masks, next_states=None, dones=None):
        if isinstance(states, list):
            if next_states is not None and len(next_states) > 0:
                for s,a,r,pi,am, n_s,d in zip(states, actions, rewards, policies, action_masks, next_states, dones):
                    self._push_one(s, a, r, pi, am, n_s, d)
            else:
                for s,a,r, pi, am in zip(states, actions, rewards, policies, action_masks):
                    self._push_one(s, a, r, pi, am)
        else:
            self._push_one(states, actions, rewards, policies, action_masks, next_states, dones)

    def sample(self, batch_size):
        if batch_size > len(self.memory):
            batch_size = len(self.memory)
        transitions = random.sample(self.memory, batch_size)
        batch = Experience(*zip(*transitions))

        # reset the memory
        self.memory = []
        self.position = 0
        return batch

    def __len__(self):
        return len(self.memory)
