"""
PyTorch Dataset for loading expert demonstrations from HDF5.

Loads the entire HDF5 into memory at construction time (datasets are small,
< 100 MB typical). Each sample is a single (obs, action) pair for frame-level
supervised learning (BC). Episode boundaries are tracked but not exposed by
default; trajectory-level access can be added later for sequence models.
"""

import numpy as np
import h5py
import torch
from torch.utils.data import Dataset


class DemoDataset(Dataset):
    """
    Frame-level dataset of expert demonstrations.

    Each item is a dict with:
        - obs:        (obs_dim,) float32 tensor
        - action:     (action_dim,) float32 tensor
        - reward:     scalar float32 (unused by BC, useful for SAC later)
        - next_obs:   (obs_dim,) float32 tensor (useful for SAC later)
        - done:       scalar bool tensor

    Attributes:
        obs_dim, action_dim: integers describing tensor shapes
        num_frames:          total number of frames (= len(self))
        num_episodes:        number of source trajectories
        episode_starts:      (num_episodes,) int64, frame index where each episode starts
        episode_lengths:     (num_episodes,) int64
    """

    def __init__(self, hdf5_path):
        with h5py.File(hdf5_path, "r") as f:
            # Load full arrays into memory
            self.obs = f["obs"][:].astype(np.float32)
            self.action = f["action"][:].astype(np.float32)
            self.reward = f["reward"][:].astype(np.float32)
            self.next_obs = f["next_obs"][:].astype(np.float32)
            self.done = f["done"][:].astype(bool)
            self.episode_starts = f["episode_starts"][:].astype(np.int64)
            self.episode_lengths = f["episode_lengths"][:].astype(np.int64)

            # Metadata
            self.obs_dim = int(f["metadata"].attrs["obs_dim"])
            self.action_dim = int(f["metadata"].attrs["action_dim"])
            self.num_episodes = int(f["metadata"].attrs["num_episodes"])
            self.num_frames = int(f["metadata"].attrs["total_steps"])

        # Sanity checks
        assert self.obs.shape == (self.num_frames, self.obs_dim)
        assert self.action.shape == (self.num_frames, self.action_dim)

    def __len__(self):
        return self.num_frames

    def __getitem__(self, idx):
        return {
            "obs": torch.from_numpy(self.obs[idx]),
            "action": torch.from_numpy(self.action[idx]),
            "reward": torch.tensor(self.reward[idx], dtype=torch.float32),
            "next_obs": torch.from_numpy(self.next_obs[idx]),
            "done": torch.tensor(self.done[idx], dtype=torch.bool),
        }

    def get_episode(self, episode_idx):
        """
        Return all frames from a specific episode as a dict of arrays.
        Useful for trajectory-level analysis (e.g., computing returns,
        visualizing rollouts).
        """
        start = int(self.episode_starts[episode_idx])
        length = int(self.episode_lengths[episode_idx])
        end = start + length
        return {
            "obs": self.obs[start:end],
            "action": self.action[start:end],
            "reward": self.reward[start:end],
            "next_obs": self.next_obs[start:end],
            "done": self.done[start:end],
        }