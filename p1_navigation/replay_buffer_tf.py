import numpy as np
import random
from collections import namedtuple, deque



class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.

        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)

        return (states, actions, rewards, next_states, dones)


class SumTree:
    """a binary tree data structure where the parent’s value is the sum of its children"""

    def __init__(self, capacity):
        """
        Tree structure and array storage:
        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for experiences
        Array type for storing:
        [0, 1, 2, 3, 4, 5, 6]
        [--------------Parent nodes-------------][-------leaves to recode priority-------]
                    size: capacity - 1                       size: capacity
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data = np.zeros(capacity, dtype=object)
        self.idx_data = 0
        self.n_entries = 0

    def __len__(self):
        return self.n_entries

    # update to the root node
    def _propagate(self, idx, change):
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    # find sample on leaf node
    def _retrieve(self, idx, s):
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        return self.tree[0]

    # store priority and experience
    def add(self, p, data):
        idx = self.idx_data + self.capacity - 1
        self.data[self.idx_data] = data
        self.update(idx, p)
        self.idx_data = (self.idx_data + 1) % self.capacity
        if self.n_entries < self.capacity:
            self.n_entries += 1

    # update priority
    def update(self, idx, p):
        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    # get priority and experience
    def get(self, s):
        idx = self._retrieve(0, s)
        idx_data = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[idx_data])


# prioritized replay buffer to store experiences in SumTree
class PrioritizedReplayBuffer:
    epsilon = 0.01
    alpha = 0.6
    beta = 0.4
    beta_increment_per_sampling = 1e-3

    def __init__(self, capacity, batch_size):
        self.capacity = capacity
        self.batch_size = batch_size
        self.tree = SumTree(capacity)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def __len__(self):
        return len(self.tree)

    def _get_priority(self, error):
        return (np.abs(error) + self.epsilon) ** self.alpha

    def add(self, state, action, reward, next_state, done, error):
        e = self.experience(state, action, reward, next_state, done)
        p = self._get_priority(error)
        self.tree.add(p, e)

    def sample(self):
        experiences, idx_tree, priorities = [], [], []
        segment = self.tree.total() / self.batch_size
        self.beta = np.min([self.beta + self.beta_increment_per_sampling, 1.0])
        for i in range(self.batch_size):
            s = random.uniform(i * segment, (i + 1) * segment)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            experiences.append(data)
            idx_tree.append(idx)
        sampling_probabilities = np.array(priorities) / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        # collect experience components
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.vstack([e.action for e in experiences if e is not None])
        rewards = np.vstack([e.reward for e in experiences if e is not None])
        next_states = np.vstack([e.next_state for e in experiences if e is not None])
        dones = np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)
        return (states, actions, rewards, next_states, dones), idx_tree, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)