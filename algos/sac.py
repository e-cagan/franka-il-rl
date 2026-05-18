"""
SAC (Soft Actor-Critic) trainer.

Off-policy actor-critic algorithm with:
  - Twin Q-networks (Q1, Q2) for overestimation mitigation
  - Soft-updated target Q-networks
  - Stochastic Gaussian policy with tanh squashing
  - Automatic entropy temperature (alpha) tuning

Online training loop: each environment step adds a transition to the
replay buffer and triggers one (or more) gradient updates on critic,
actor, and alpha.

Reference:
    Haarnoja et al. (2018), "Soft Actor-Critic Algorithms and Applications"
"""

import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from networks.gaussian_policy import GaussianPolicy
from networks.q_network import QNetwork
from data_utils.replay_buffer import ReplayBuffer


class SACTrainer:
    """
    Trainer for Soft Actor-Critic.

    Args:
        env: Gymnasium-compatible environment.
        evaluator: Evaluator for periodic env evaluation.
        config: dict with hyperparameters (see below).
        logger: optional WandBLogger.
        checkpoint_dir: directory for saving checkpoints.

    Config keys (with defaults):
        Algorithm:
          gamma:             0.99    # discount factor
          tau:               0.005   # target network soft-update rate
          target_entropy:    -action_dim  # entropy target (per Haarnoja)
          init_alpha:        0.2     # initial entropy temperature
        Networks:
          obs_dim:           28
          action_dim:        4
          policy_hidden:     [256, 256]
          q_hidden:          [256, 256]
        Optimization:
          lr_actor:          3.0e-4
          lr_critic:         3.0e-4
          lr_alpha:          3.0e-4
          batch_size:        256
        Training:
          total_env_steps:   100000
          warmup_steps:      1000    # random actions before training starts
          updates_per_step:  1       # gradient updates per env step
          eval_every_steps:  5000
          eval_episodes:     20
        Buffer:
          buffer_capacity:   200000
        Hardware:
          device:            "cuda" or "cpu"
    """

    def __init__(self, env, evaluator, config, logger=None,
                 checkpoint_dir="data/checkpoints/sac"):
        self.env = env
        self.evaluator = evaluator
        self.config = config
        self.logger = logger

        # Resolve hyperparameters
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 256)
        self.total_env_steps = config.get("total_env_steps", 100_000)
        self.warmup_steps = config.get("warmup_steps", 1000)
        self.updates_per_step = config.get("updates_per_step", 1)
        self.eval_every_steps = config.get("eval_every_steps", 5000)

        self.obs_dim = config.get("obs_dim", 28)
        self.action_dim = config.get("action_dim", 4)

        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )

        # --- Networks ---
        self.policy = GaussianPolicy(
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            hidden_sizes=tuple(config.get("policy_hidden", [256, 256])),
        ).to(self.device)

        q_hidden = tuple(config.get("q_hidden", [256, 256]))
        self.q1 = QNetwork(self.obs_dim, self.action_dim, q_hidden).to(self.device)
        self.q2 = QNetwork(self.obs_dim, self.action_dim, q_hidden).to(self.device)

        # Target Q-networks: deep copies, weights frozen from autograd
        self.q1_target = copy.deepcopy(self.q1)
        self.q2_target = copy.deepcopy(self.q2)
        for p in self.q1_target.parameters():
            p.requires_grad = False
        for p in self.q2_target.parameters():
            p.requires_grad = False

        # --- Optimizers ---
        lr_actor = config.get("lr_actor", 3e-4)
        lr_critic = config.get("lr_critic", 3e-4)
        lr_alpha = config.get("lr_alpha", 3e-4)

        self.actor_optim = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)
        self.q1_optim = torch.optim.Adam(self.q1.parameters(), lr=lr_critic)
        self.q2_optim = torch.optim.Adam(self.q2.parameters(), lr=lr_critic)

        # --- Entropy temperature (alpha) ---
        # Learn log_alpha for stability (alpha = exp(log_alpha) > 0 always)
        init_alpha = config.get("init_alpha", 0.2)
        self.log_alpha = torch.tensor(
            np.log(init_alpha), dtype=torch.float32,
            device=self.device, requires_grad=True,
        )
        self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        # Target entropy: -|A| is the Haarnoja recommendation
        self.target_entropy = config.get("target_entropy", -float(self.action_dim))

        # --- Replay buffer ---
        self.buffer = ReplayBuffer(
            capacity=config.get("buffer_capacity", 200_000),
            obs_dim=self.obs_dim,
            action_dim=self.action_dim,
            device=self.device,
        )

        # --- Bookkeeping ---
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.global_step = 0
        self.gradient_updates = 0
        self.best_success_rate = -1.0

    @property
    def alpha(self):
        """Current entropy temperature (exponentiated log_alpha)."""
        return self.log_alpha.exp().detach()

    def _soft_update_targets(self):
        """
        Polyak averaging: target ← τ·source + (1-τ)·target
        """
        with torch.no_grad():
            for tp, sp in zip(self.q1_target.parameters(),
                              self.q1.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * sp.data)
            for tp, sp in zip(self.q2_target.parameters(),
                              self.q2.parameters()):
                tp.data.mul_(1.0 - self.tau)
                tp.data.add_(self.tau * sp.data)
    
    def update_critic(self, batch):
        """
        Update Q1 and Q2 by minimizing TD error:
            target = r + γ·(1-done)·[min(Q1_t, Q2_t)(s', a') - α·log_π(a'|s')]
            loss   = MSE(Q(s, a), target)

        where a' is sampled fresh from current policy at s' (off-policy
        SAC: target uses the policy that we're currently learning, not
        the action stored in the buffer).
        """
        obs = batch["obs"]
        action = batch["action"]
        reward = batch["reward"].unsqueeze(-1)         # (B,) -> (B, 1)
        next_obs = batch["next_obs"]
        done = batch["done"].unsqueeze(-1)             # (B,) -> (B, 1)

        # --- Compute TD target (no gradients into target) ---
        with torch.no_grad():
            next_action, next_log_prob, _ = self.policy.sample(next_obs)
            q1_t = self.q1_target(next_obs, next_action)
            q2_t = self.q2_target(next_obs, next_action)
            min_q_t = torch.min(q1_t, q2_t)
            # Soft Bellman target: subtract entropy term
            target_v = min_q_t - self.alpha * next_log_prob
            target_q = reward + self.gamma * (1.0 - done) * target_v

        # --- Q1 loss + step ---
        q1_pred = self.q1(obs, action)
        q1_loss = F.mse_loss(q1_pred, target_q)
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        # --- Q2 loss + step ---
        q2_pred = self.q2(obs, action)
        q2_loss = F.mse_loss(q2_pred, target_q)
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "q1_mean": q1_pred.mean().item(),
            "q2_mean": q2_pred.mean().item(),
            "target_q_mean": target_q.mean().item(),
        }

    def update_actor(self, batch):
        """
        Update policy by maximizing:
            E[ min(Q1, Q2)(s, π(s)) - α·log π(π(s)|s) ]

        Equivalently, minimize:
            loss = (α·log_prob - min_Q).mean()

        Sample a NEW action from current policy (reparameterized) so
        gradients flow through the policy. The buffer's stored action
        is not used here.
        """
        obs = batch["obs"]
        new_action, log_prob, _ = self.policy.sample(obs)

        q1_val = self.q1(obs, new_action)
        q2_val = self.q2(obs, new_action)
        min_q = torch.min(q1_val, q2_val)

        actor_loss = (self.alpha * log_prob - min_q).mean()

        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        # Return log_prob.detach() for use in alpha update (same batch)
        return {
            "actor_loss": actor_loss.item(),
            "log_prob_mean": log_prob.mean().item(),
            "entropy_mean": (-log_prob).mean().item(),
        }, log_prob.detach()

    def update_alpha(self, log_prob_detached):
        """
        Update entropy temperature α.

        Sign convention here: target_entropy is stored as a NEGATIVE
        number (e.g., -action_dim = -4) following one common SAC formulation.
        Effective "desired entropy" is -target_entropy (= +4).

        Loss derivation:
            entropy(s) = -log_prob(s).mean()
            We want entropy ≈ -target_entropy.
            Gradient sign:
              entropy < -target → need MORE exploration → α should INCREASE
              entropy > -target → need LESS exploration → α should DECREASE

        Equivalently, with log_prob_mean and target_entropy (negative):
            alpha_loss = log_alpha · (log_prob_mean - target_entropy).detach()
        Gradient w.r.t. log_alpha: (log_prob_mean - target_entropy)
            entropy high (log_prob very negative): log_prob_mean - target < 0
                → grad < 0 → log_alpha decreases → α decreases ✓
            entropy low (log_prob less negative): log_prob_mean - target > 0
                → grad > 0 → log_alpha increases → α increases ✓
        """
        log_prob_mean = log_prob_detached.mean()
        alpha_loss = -(self.log_alpha
                      * (log_prob_mean - self.target_entropy).detach())

        self.alpha_optim.zero_grad()
        alpha_loss.backward()
        self.alpha_optim.step()

        entropy = -log_prob_mean.item()
        return {
            "alpha_loss": alpha_loss.item(),
            "alpha": self.alpha.item(),
            "entropy": entropy,
        }