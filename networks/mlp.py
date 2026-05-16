"""
Deterministic MLP policy for behavioral cloning.

Architecture:
    obs (obs_dim) -> Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU
    -> Linear -> tanh -> action (action_dim) in [-1, 1]

Output is tanh-squashed because action spaces in our envs are [-1, 1].
"""

import torch
import torch.nn as nn


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron mapping observations to deterministic actions.

    Args:
        obs_dim: input observation dimension.
        action_dim: output action dimension.
        hidden_sizes: tuple of hidden layer widths.
                      Default (256, 256, 256) for ~200k params.
        activation: nonlinearity between hidden layers. Default ReLU.
    """

    def __init__(self, obs_dim, action_dim, hidden_sizes=(256, 256, 256),
                 activation=nn.ReLU):
        super().__init__()

        layers = []
        in_dim = obs_dim
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(activation())
            in_dim = h
        # Final layer maps to action_dim, no activation here
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)
        self.obs_dim = obs_dim
        self.action_dim = action_dim

    def forward(self, obs):
        """
        Args:
            obs: (batch, obs_dim) tensor.
        Returns:
            action: (batch, action_dim) tensor, squashed to [-1, 1] via tanh.
        """
        raw = self.net(obs)
        return torch.tanh(raw)

    def act(self, obs_numpy):
        """
        Inference helper for use during environment rollouts.

        Args:
            obs_numpy: (obs_dim,) numpy array, single observation.
        Returns:
            action: (action_dim,) numpy array.
        """
        self.eval()
        with torch.no_grad():
            obs_tensor = torch.from_numpy(obs_numpy).float().unsqueeze(0)
            action_tensor = self.forward(obs_tensor)
            action = action_tensor.squeeze(0).cpu().numpy()
        return action