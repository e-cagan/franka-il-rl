"""
Q-network (critic) for SAC.

Takes (obs, action) concatenated and outputs a scalar Q-value estimate.
SAC uses two Q-networks (Twin Q) to mitigate overestimation bias; this
file defines a single Q-network class — the SAC trainer will instantiate
two of them with different random initializations.
"""

import torch
import torch.nn as nn


class QNetwork(nn.Module):
    """
    Q(s, a) approximator.

    Args:
        obs_dim: observation dimension.
        action_dim: action dimension.
        hidden_sizes: tuple of hidden widths. Default (256, 256).

    Forward:
        obs:    (batch, obs_dim)
        action: (batch, action_dim)
        returns: (batch, 1) Q-value
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim + action_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.net(x)