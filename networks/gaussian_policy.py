"""
Gaussian policy with tanh squashing, for SAC.

Architecture:
    obs -> MLP trunk -> (mean, log_std)
    raw_action = Normal(mean, exp(log_std)).rsample()
    action = tanh(raw_action)        # squash to [-1, 1]

The log_prob of the squashed action accounts for the tanh Jacobian
correction, using a numerically stable form.
"""

import math
import torch
import torch.nn as nn


LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class GaussianPolicy(nn.Module):
    """
    Tanh-squashed Gaussian policy.

    Args:
        obs_dim: input observation dimension.
        action_dim: output action dimension.
        hidden_sizes: tuple of trunk hidden widths. Default (256, 256).

    Outputs (forward):
        mean:    (batch, action_dim)
        log_std: (batch, action_dim), clipped to [LOG_STD_MIN, LOG_STD_MAX]

    Sampling:
        sample(obs) -> (action, log_prob, mean)
            action:   tanh-squashed sample
            log_prob: log π(action | obs), with tanh correction
            mean:     deterministic mean action (for eval rollouts)
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256)):
        super().__init__()
        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.trunk = nn.Sequential(*layers)

        # Two heads: mean and log_std
        self.mean_head = nn.Linear(in_dim, action_dim)
        self.log_std_head = nn.Linear(in_dim, action_dim)

        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs):
        """Return (mean, log_std). log_std is clipped for stability."""
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        """
        Sample an action with reparameterization (gradients flow through).

        Returns:
            action:   (batch, action_dim), in [-1, 1]
            log_prob: (batch, 1), log probability of action under policy
            mean_action: (batch, action_dim), tanh(mean) for eval
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()

        # Reparameterized sample (preserves gradient)
        normal = torch.distributions.Normal(mean, std)
        raw_action = normal.rsample()              # u
        action = torch.tanh(raw_action)            # a = tanh(u)

        # Log-prob with tanh correction
        # log π(a) = log p(u) - sum(log(1 - tanh(u)^2))
        # Using numerically stable form:
        # log(1 - tanh(u)^2) = 2 * (log(2) - u - softplus(-2u))
        log_prob_u = normal.log_prob(raw_action)
        log_correction = 2.0 * (math.log(2.0) - raw_action
                                - torch.nn.functional.softplus(-2.0 * raw_action))
        log_prob = (log_prob_u - log_correction).sum(dim=-1, keepdim=True)

        mean_action = torch.tanh(mean)
        return action, log_prob, mean_action

    def act(self, obs_numpy, deterministic=False):
        """
        Inference helper for env rollouts.

        Args:
            obs_numpy: (obs_dim,) numpy array.
            deterministic: if True, return tanh(mean) (for eval).
                          If False, sample (for training rollouts).
        """
        import numpy as np
        self.eval()
        with torch.no_grad():
            device = next(self.parameters()).device
            obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0).to(device)
            if deterministic:
                mean, _ = self.forward(obs_tensor)
                action = torch.tanh(mean)
            else:
                action, _, _ = self.sample(obs_tensor)
            return action.squeeze(0).cpu().numpy()