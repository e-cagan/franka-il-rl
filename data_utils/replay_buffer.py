"""
FIFO replay buffer for SAC.

Stores transitions (obs, action, reward, next_obs, done) in pre-allocated
numpy arrays. Uniform random sampling. Single-threaded; no priority
sampling, no n-step returns. Designed for sub-1M-transition workloads
on CPU.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-capacity FIFO buffer with uniform sampling.

    Args:
        capacity: maximum number of transitions to store.
        obs_dim: observation dimension.
        action_dim: action dimension.
        device: torch device for returned batch tensors (e.g. "cuda").

    Stored data is float32 (numpy on CPU). On sample(), batches are
    converted to torch tensors and moved to the requested device.
    """

    def __init__(self, capacity, obs_dim, action_dim, device="cpu"):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device

        # Pre-allocated storage
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.action = np.zeros((capacity, action_dim), dtype=np.float32)
        self.reward = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros(capacity, dtype=np.float32)  # float for masking

        self.pos = 0       # next write index
        self.size = 0      # current count (≤ capacity)

    def add(self, obs, action, reward, next_obs, done):
        """Append a single transition. Wraps around when full (FIFO)."""
        self.obs[self.pos] = obs
        self.action[self.pos] = action
        self.reward[self.pos] = reward
        self.next_obs[self.pos] = next_obs
        self.done[self.pos] = float(done)

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        """
        Uniform random sample. Returns dict of torch tensors on self.device.

        Raises:
            ValueError if size < batch_size.
        """
        if self.size < batch_size:
            raise ValueError(
                f"Buffer has {self.size} transitions, cannot sample {batch_size}"
            )

        indices = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs":      torch.from_numpy(self.obs[indices]).to(self.device),
            "action":   torch.from_numpy(self.action[indices]).to(self.device),
            "reward":   torch.from_numpy(self.reward[indices]).to(self.device),
            "next_obs": torch.from_numpy(self.next_obs[indices]).to(self.device),
            "done":     torch.from_numpy(self.done[indices]).to(self.device),
        }

    def __len__(self):
        return self.size