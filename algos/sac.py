"""
SAC (Soft Actor-Critic) trainer.

Off-policy actor-critic algorithm with:
  - Twin Q-networks (Q1, Q2) for overestimation mitigation
  - Soft-updated target Q-networks
  - Stochastic Gaussian policy with tanh squashing
  - Automatic entropy temperature (alpha) tuning
  - Optional HER (Hindsight Experience Replay) for sparse goal-conditioned tasks
  - Optional demo pre-fill (DAPG-style: expert trajectories seeded into the
    replay buffer to bootstrap exploration in hard sparse-reward tasks)

Online training loop: each environment step adds a transition to the
replay buffer and triggers one (or more) gradient updates on critic,
actor, and alpha.

References:
    Haarnoja et al. (2018), "Soft Actor-Critic Algorithms and Applications"
    Andrychowicz et al. (2017), "Hindsight Experience Replay"
    Rajeswaran et al. (2018), "Learning Complex Dexterous Manipulation
        with Deep RL and Demonstrations" (DAPG)
"""

import copy
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

from networks.gaussian_policy import GaussianPolicy
from networks.q_network import QNetwork
from data_utils.replay_buffer import ReplayBuffer
from data_utils.her_replay_buffer import HERReplayBuffer
from data_utils.demo_dataset import DemoDataset


class SACTrainer:
    """
    Trainer for Soft Actor-Critic.

    Args:
        env: Gymnasium-compatible environment. For HER mode, env must
             expose `info["achieved_goal"]` on reset/step, a
             `compute_reward(achieved_goal, desired_goal, info)` method,
             and an `extract_achieved_goal(flat_obs)` helper (the
             FetchPickPlaceWrapper provides all three).
        evaluator: Evaluator for periodic env evaluation.
        config: dict with hyperparameters (see below).
        logger: optional WandBLogger.
        checkpoint_dir: directory for saving checkpoints.

    Config keys (with defaults):
        Algorithm:
          gamma:             0.99    # discount factor
          tau:               0.005   # target network soft-update rate
          target_entropy:    -action_dim/2  # entropy target.
                                            # NB: -|A| (Haarnoja default) is
                                            # unreachable for tanh-squashed
                                            # Gaussians since H_max = |A|*log(2).
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
          use_her:           False   # enable hindsight experience replay
          her_k:             4       # relabels per original transition
          her_strategy:      "future"
          goal_dim:          3       # achieved_goal dimension (HER only)
          demo_path:         None    # if set, pre-fill buffer with these
                                     # demos at fit() start. Requires use_her
                                     # for now (relabeled demos drive learning).
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
        # Target entropy: default to -|A|/2. Note that -|A| (Haarnoja
        # heuristic) is unreachable for a tanh-squashed Gaussian since
        # achievable entropy is bounded by |A|*log(2).
        self.target_entropy = config.get(
            "target_entropy", -0.5 * float(self.action_dim)
        )

        # --- Replay buffer (HER or vanilla) ---
        self.use_her = bool(config.get("use_her", False))
        if self.use_her:
            goal_dim = config.get(
                "goal_dim",
                getattr(env, "goal_dim", 3),
            )
            goal_slice = getattr(
                env, "goal_slice",
                slice(self.obs_dim - goal_dim, self.obs_dim),
            )
            self.buffer = HERReplayBuffer(
                capacity=config.get("buffer_capacity", 1_000_000),
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                goal_dim=goal_dim,
                goal_slice=goal_slice,
                compute_reward_fn=env.compute_reward,
                k=config.get("her_k", 4),
                strategy=config.get("her_strategy", "future"),
                device=self.device,
            )
        else:
            self.buffer = ReplayBuffer(
                capacity=config.get("buffer_capacity", 200_000),
                obs_dim=self.obs_dim,
                action_dim=self.action_dim,
                device=self.device,
            )

        # Demo pre-fill path (loaded at fit() start, not __init__, so the
        # constructor stays cheap)
        self.demo_path = config.get("demo_path")

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

    def _prefill_from_demos(self, demo_path):
        """
        Pre-fill the HER buffer with expert demonstrations.

        Each demo episode is staged into the buffer's episode buffer and
        then flushed, producing T originals + (T-1)*k HER relabels per
        episode. The expert's actions are stored as-is; SAC's off-policy
        nature means the critic can bootstrap from demonstration tuples
        even though the actor is trained against fresh on-policy actions.

        This is the DAPG-style "demos in buffer" technique (Rajeswaran 2018),
        adapted for HER-augmented SAC. It addresses the FetchPickAndPlace
        exploration bottleneck: random policy never grasps the cube, so
        achieved_goals are constant within an episode and HER relabels are
        trivial (always at the initial cube position). Demos provide the
        diverse achieved_goal trajectories that HER needs to be useful.

        Args:
            demo_path: HDF5 file path; expected layout matches DemoDataset.
        """
        if not self.use_her:
            raise ValueError(
                "Demo pre-fill currently requires use_her=True. The HER "
                "buffer's flush_episode() generates the relabeled "
                "transitions that make demos useful for sparse-reward SAC."
            )

        print(f"Loading demos from {demo_path}...")
        dataset = DemoDataset(demo_path)

        for ep_idx in range(dataset.num_episodes):
            start = int(dataset.episode_starts[ep_idx])
            length = int(dataset.episode_lengths[ep_idx])
            for t in range(length):
                idx = start + t
                # Demos don't store info["achieved_goal"]; recover it from
                # the next_obs cube-position slice.
                next_ag = self.env.extract_achieved_goal(dataset.next_obs[idx])
                self.buffer.add(
                    obs=dataset.obs[idx],
                    action=dataset.action[idx],
                    reward=float(dataset.reward[idx]),
                    next_obs=dataset.next_obs[idx],
                    done=float(dataset.done[idx]),
                    next_achieved_goal=next_ag,
                )
            self.buffer.flush_episode()

        print(f"  Pre-filled buffer with {dataset.num_episodes} demo episodes "
              f"-> {len(self.buffer)} transitions (originals + HER relabels).")

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

        return {
            "actor_loss": actor_loss.item(),
            "log_prob_mean": log_prob.mean().item(),
            "entropy_mean": (-log_prob).mean().item(),
        }, log_prob.detach()

    def update_alpha(self, log_prob_detached):
        """
        Update entropy temperature α.

        target_entropy is stored as a NEGATIVE number (e.g., -2 for 4-D);
        effective "desired entropy" is -target_entropy (= +2).

        Loss form:
            alpha_loss = -log_alpha · (log_prob_mean - target_entropy).detach()
        Gradient w.r.t. log_alpha: -(log_prob_mean - target_entropy)
            entropy high (log_prob very negative): log_prob_mean - target < 0
                → grad > 0 → log_alpha decreases → α decreases ✓
            entropy low (log_prob less negative): log_prob_mean - target > 0
                → grad < 0 → log_alpha increases → α increases ✓
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

    def select_action(self, obs_numpy, deterministic=False):
        """Helper: convert obs to action via current policy."""
        return self.policy.act(obs_numpy, deterministic=deterministic)

    def evaluate(self):
        """Run deterministic policy in env, return metrics."""
        def policy_fn(obs):
            return self.policy.act(obs, deterministic=True)
        return self.evaluator.evaluate(policy_fn)

    def save_checkpoint(self, name):
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "q1_state_dict": self.q1.state_dict(),
            "q2_state_dict": self.q2.state_dict(),
            "q1_target_state_dict": self.q1_target.state_dict(),
            "q2_target_state_dict": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optim_state": self.actor_optim.state_dict(),
            "q1_optim_state": self.q1_optim.state_dict(),
            "q2_optim_state": self.q2_optim.state_dict(),
            "alpha_optim_state": self.alpha_optim.state_dict(),
            "config": self.config,
            "global_step": self.global_step,
            "best_success_rate": self.best_success_rate,
        }, path)
        return path

    def fit(self):
        """Main SAC training loop."""
        # --- 0. Optional demo pre-fill (DAPG-style) ---
        if self.demo_path:
            self._prefill_from_demos(self.demo_path)

        her_str = " + HER" if self.use_her else ""
        demo_str = " + demos" if self.demo_path else ""
        print(f"SAC{her_str}{demo_str}: {self.total_env_steps} env steps "
              f"(warmup={self.warmup_steps}, "
              f"updates_per_step={self.updates_per_step}, "
              f"device={self.device})")

        obs, info = self.env.reset(seed=self.config.get("seed", 42))
        episode_return = 0.0
        episode_length = 0
        episode_count = 0
        recent_returns = []  # last 10 episode returns
        recent_successes = []  # last 10 episode successes

        # Cumulative metrics for the current step
        step_metrics = {}

        for step in range(self.total_env_steps):
            self.global_step = step

            # --- 1. Select action ---
            if step < self.warmup_steps:
                # Random action during warmup
                action = self.env.action_space.sample().astype(np.float32)
            else:
                action = self.policy.act(obs, deterministic=False)

            # --- 2. Env step ---
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated
            # Treat truncation as terminal (FetchPickAndPlace-v4 never emits
            # terminated=True; episodes end via truncated=True at step 50).
            store_done = float(terminated or truncated)

            # --- 3. Buffer add ---
            if self.use_her:
                next_achieved_goal = info["achieved_goal"]
                self.buffer.add(obs, action, reward, next_obs,
                                store_done, next_achieved_goal)
            else:
                self.buffer.add(obs, action, reward, next_obs, store_done)

            episode_return += reward
            episode_length += 1

            obs = next_obs

            if done:
                # HER: flush the staged episode (writes originals + relabels)
                if self.use_her:
                    self.buffer.flush_episode()

                ep_success = float(info.get("is_success", 0.0) > 0.5)
                recent_returns.append(episode_return)
                recent_successes.append(ep_success)
                if len(recent_returns) > 10:
                    recent_returns.pop(0)
                    recent_successes.pop(0)
                episode_count += 1
                episode_return = 0.0
                episode_length = 0
                obs, info = self.env.reset(seed=self.config.get("seed", 42)
                                            + episode_count)

            # --- 4. Gradient updates (after warmup) ---
            if step >= self.warmup_steps and len(self.buffer) >= self.batch_size:
                for _ in range(self.updates_per_step):
                    batch = self.buffer.sample(self.batch_size)
                    critic_m = self.update_critic(batch)
                    actor_m, log_prob_d = self.update_actor(batch)
                    alpha_m = self.update_alpha(log_prob_d)
                    self._soft_update_targets()
                    self.gradient_updates += 1
                    step_metrics = {**critic_m, **actor_m, **alpha_m}

            # --- 5. Periodic eval ---
            if (step + 1) % self.eval_every_steps == 0 and step >= self.warmup_steps:
                eval_metrics = self.evaluate()
                success = eval_metrics["success_rate"]

                print(f"[step {step+1:6d}] "
                      f"eval_success={success:.2%}  "
                      f"eval_return={eval_metrics['mean_return']:.1f}  "
                      f"alpha={self.alpha.item():.3f}  "
                      f"q1_loss={step_metrics.get('q1_loss', 0):.3f}  "
                      f"buffer={len(self.buffer)}")

                if success > self.best_success_rate:
                    self.best_success_rate = success
                    self.save_checkpoint("best_success")

                # Log to W&B
                if self.logger is not None:
                    train_returns = float(np.mean(recent_returns)) if recent_returns else 0.0
                    train_success = float(np.mean(recent_successes)) if recent_successes else 0.0
                    self.logger.log({
                        "train/episode_return_recent10": train_returns,
                        "train/episode_success_recent10": train_success,
                        "eval/success_rate": success,
                        "eval/mean_return": eval_metrics["mean_return"],
                        "sac/alpha": self.alpha.item(),
                        "sac/log_alpha_raw": self.log_alpha.item(),
                        "sac/entropy": -step_metrics.get("log_prob_mean", 0),
                        "sac/q1_loss": step_metrics.get("q1_loss", 0),
                        "sac/q2_loss": step_metrics.get("q2_loss", 0),
                        "sac/actor_loss": step_metrics.get("actor_loss", 0),
                        "sac/q1_mean": step_metrics.get("q1_mean", 0),
                        "sac/target_q_mean": step_metrics.get("target_q_mean", 0),
                        "sac/gradient_updates": self.gradient_updates,
                        "sac/buffer_size": len(self.buffer),
                    }, step=step + 1)

            # Light progress every 1000 steps during warmup or pre-eval
            elif (step + 1) % 1000 == 0 and step < self.warmup_steps:
                print(f"[step {step+1:6d}] warmup, buffer={len(self.buffer)}")

        # Final checkpoint
        self.save_checkpoint("last")
        print(f"\nSAC training complete.")
        print(f"  Best eval success: {self.best_success_rate:.2%}")
        print(f"  Total gradient updates: {self.gradient_updates}")
        print(f"  Checkpoints in: {self.checkpoint_dir}")