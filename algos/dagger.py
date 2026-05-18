"""
DAgger (Dataset Aggregation) trainer.

Iteratively:
  1. Roll out mixed policy (β·expert + (1-β)·policy) to collect states
  2. Query expert for the correct action at each visited state
  3. Aggregate (state, expert_action) pairs into the training dataset
  4. Retrain policy on aggregated dataset (BC-style)

The aggregated dataset is capped at a fixed size; oldest episodes are
evicted FIFO when new ones are added beyond the cap.
"""

import os
from pathlib import Path
from collections import deque
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class AggregatedDataset:
    """
    A growing, FIFO-capped dataset of (obs, expert_action) pairs.

    Internally stores frames per episode so that eviction can drop whole
    episodes (preserving trajectory boundaries for any future analysis,
    even though BC training itself is frame-level).

    Args:
        cap_episodes: maximum number of episodes to retain.
    """

    def __init__(self, cap_episodes):
        self.cap_episodes = cap_episodes
        # Each entry: dict with "obs" (T, obs_dim), "action" (T, action_dim)
        self.episodes = deque()

    def add_episode(self, obs_array, action_array):
        """Append one episode. If over cap, evict the oldest."""
        self.episodes.append({
            "obs": np.asarray(obs_array, dtype=np.float32),
            "action": np.asarray(action_array, dtype=np.float32),
        })
        while len(self.episodes) > self.cap_episodes:
            self.episodes.popleft()

    def add_from_hdf5(self, path):
        """Seed the dataset from an existing demonstrations HDF5 file."""
        import h5py
        with h5py.File(path, "r") as f:
            starts = f["episode_starts"][:]
            lengths = f["episode_lengths"][:]
            obs = f["obs"][:]
            action = f["action"][:]
            for s, L in zip(starts, lengths):
                self.add_episode(obs[s:s + L], action[s:s + L])

    def num_episodes(self):
        return len(self.episodes)

    def num_frames(self):
        return sum(len(e["obs"]) for e in self.episodes)

    def as_tensors(self):
        """Concatenate all episodes into flat (frames, dim) tensors."""
        all_obs = np.concatenate([e["obs"] for e in self.episodes], axis=0)
        all_act = np.concatenate([e["action"] for e in self.episodes], axis=0)
        return torch.from_numpy(all_obs), torch.from_numpy(all_act)

    def as_dataloader(self, batch_size, shuffle=True, pin_memory=False):
        obs_t, act_t = self.as_tensors()
        ds = TensorDataset(obs_t, act_t)
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          pin_memory=pin_memory)


def mixed_rollout(env, policy, expert, beta, num_episodes,
                  expert_reset_callback=None, seed_start=20000):
    """
    Collect rollouts using a mixed policy:
        executed_action = expert.act()       with prob β
        executed_action = policy.act(obs)    with prob 1-β

    Regardless of which action is executed, we ALWAYS record the
    expert's action as the label for every visited state. This is what
    DAgger trains on: "what would the expert do here?"

    Args:
        env:     environment instance (Gymnasium-compatible).
        policy:  current learner policy with .act(obs_numpy) method.
        expert:  scripted expert with .act() and .reset() methods.
        beta:    mixing coefficient, 0 ≤ β ≤ 1.
        num_episodes: how many episodes to roll out.
        expert_reset_callback: invoked after env.reset() each episode
                               (the FetchExpert needs this; pass expert.reset).
        seed_start: starting seed for env.reset, for reproducibility.

    Returns:
        list of dicts, one per episode, with keys "obs", "action", "success".
        "action" is the expert label, not the executed action.
    """
    episodes = []
    for ep in range(num_episodes):
        obs, info = env.reset(seed=seed_start + ep)
        if expert_reset_callback is not None:
            expert_reset_callback()

        ep_obs = []
        ep_expert_actions = []
        ep_success = False

        while True:
            # Always query expert for the label
            expert_action = expert.act()
            ep_obs.append(obs.copy())
            ep_expert_actions.append(expert_action.copy())

            # Decide which action to execute
            if np.random.rand() < beta:
                exec_action = expert_action
            else:
                exec_action = policy.act(obs)

            obs, reward, terminated, truncated, info = env.step(exec_action)

            if info.get("is_success", 0.0) > 0.5:
                ep_success = True
            if terminated or truncated:
                break

        episodes.append({
            "obs": np.asarray(ep_obs, dtype=np.float32),
            "action": np.asarray(ep_expert_actions, dtype=np.float32),
            "success": ep_success,
        })

    return episodes


class DAggerTrainer:
    """
    DAgger trainer that wraps an MLPPolicy and iteratively aggregates
    expert-labeled rollouts.

    Algorithm:
        For each DAgger iteration i in 0..N-1:
          1. β = max(0, 1 - i/N)  (linear decay)
          2. Collect rollouts using mixed policy (β·expert + (1-β)·policy)
          3. Aggregate all visited states + expert labels into dataset
          4. Retrain policy on aggregated dataset for `inner_epochs` epochs
          5. Evaluate policy in env (every iteration)
          6. Save checkpoint

    The inner BC loop reuses the same loss (MSE) and optimizer (AdamW)
    as Week 5's BCTrainer.

    Args:
        policy:    MLPPolicy instance (already initialized).
        env:       environment for rollouts and evaluation.
        expert:    scripted expert with .act() and .reset().
        evaluator: Evaluator instance for periodic env eval.
        config:    dict with hyperparameters:
                   - num_iterations  (default 10)
                   - rollouts_per_iter (default 20)
                   - inner_epochs   (default 20, BC epochs per iter)
                   - lr, weight_decay, batch_size (BC optimizer params)
                   - dataset_cap    (default 800 episodes)
                   - initial_dataset_path (HDF5 to seed aggregated dataset)
                   - eval_episodes  (default 20, in-training eval)
                   - rollout_seed_start (default 20000, env seeds for rollouts)
        logger:    optional WandBLogger.
        checkpoint_dir: directory to save iteration checkpoints.
    """

    def __init__(self, policy, env, expert, evaluator, config,
                 logger=None, checkpoint_dir="data/checkpoints/dagger"):
        import torch.nn.functional as F

        self.policy = policy
        self.env = env
        self.expert = expert
        self.evaluator = evaluator
        self.config = config
        self.logger = logger
        self.F = F  # store reference for use in inner loop

        self.device = config.get(
            "device", "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.policy.to(self.device)

        self.num_iterations = config.get("num_iterations", 10)
        self.rollouts_per_iter = config.get("rollouts_per_iter", 20)
        self.inner_epochs = config.get("inner_epochs", 20)
        self.batch_size = config.get("batch_size", 256)
        self.eval_episodes = config.get("eval_episodes", 20)
        self.rollout_seed_start = config.get("rollout_seed_start", 20000)

        # Aggregated dataset, seeded from BC training set
        self.dataset = AggregatedDataset(
            cap_episodes=config.get("dataset_cap", 800)
        )
        seed_path = config.get("initial_dataset_path")
        if seed_path is not None:
            print(f"Seeding aggregated dataset from {seed_path}")
            self.dataset.add_from_hdf5(seed_path)
            print(f"  initial: {self.dataset.num_episodes()} episodes, "
                  f"{self.dataset.num_frames()} frames")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.global_step = 0
        self.best_success_rate = -1.0

    def compute_beta(self, iteration):
        """Linear decay: β = max(0, 1 - i/N)."""
        return max(0.0, 1.0 - iteration / self.num_iterations)

    def train_inner_bc(self):
        """Retrain policy on aggregated dataset for inner_epochs epochs."""
        loader = self.dataset.as_dataloader(
            batch_size=self.batch_size,
            shuffle=True,
            pin_memory=(self.device == "cuda"),
        )
        self.policy.train()
        epoch_mean_losses = []
        for epoch in range(self.inner_epochs):
            losses = []
            for obs, action in loader:
                obs = obs.to(self.device, non_blocking=True)
                action = action.to(self.device, non_blocking=True)
                pred = self.policy(obs)
                loss = self.F.mse_loss(pred, action)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses.append(loss.item())
                self.global_step += 1
            epoch_mean_losses.append(float(np.mean(losses)))
        return float(np.mean(epoch_mean_losses)), epoch_mean_losses[-1]

    def evaluate_in_env(self):
        """Roll out current policy with no expert mixing, return metrics."""
        self.policy.eval()
        def policy_fn(obs_np):
            return self.policy.act(obs_np)
        return self.evaluator.evaluate(policy_fn)

    def save_checkpoint(self, name):
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_success_rate": self.best_success_rate,
        }, path)
        return path

    def fit(self):
        """Main DAgger loop."""
        print(f"DAgger: {self.num_iterations} iterations, "
              f"{self.rollouts_per_iter} rollouts/iter, "
              f"{self.inner_epochs} inner BC epochs/iter")

        for it in range(self.num_iterations):
            beta = self.compute_beta(it)
            print(f"\n=== Iteration {it+1}/{self.num_iterations} "
                  f"(β={beta:.2f}) ===")

            # 1. Mixed rollout
            self.policy.eval()
            new_episodes = mixed_rollout(
                env=self.env,
                policy=self.policy,
                expert=self.expert,
                beta=beta,
                num_episodes=self.rollouts_per_iter,
                expert_reset_callback=self.expert.reset,
                seed_start=self.rollout_seed_start + it * self.rollouts_per_iter,
            )
            rollout_success = np.mean([e["success"] for e in new_episodes])

            # 2. Aggregate
            for ep in new_episodes:
                self.dataset.add_episode(ep["obs"], ep["action"])
            print(f"  Rollout success (mixed policy): {rollout_success:.2%}")
            print(f"  Dataset now: {self.dataset.num_episodes()} ep, "
                  f"{self.dataset.num_frames()} frames")

            # 3. Inner BC training
            mean_loss, last_loss = self.train_inner_bc()
            print(f"  Inner BC training mean_loss: {mean_loss:.4f} "
                  f"(last epoch: {last_loss:.4f})")

            # 4. Eval (pure policy, no expert mixing)
            eval_metrics = self.evaluate_in_env()
            success = eval_metrics["success_rate"]
            print(f"  Pure-policy success ({self.eval_episodes} ep): "
                  f"{success:.2%}, mean_return={eval_metrics['mean_return']:.1f}")

            # 5. Best-tracking + checkpoint
            if success > self.best_success_rate:
                self.best_success_rate = success
                self.save_checkpoint("best_success")

            # 6. Log
            if self.logger is not None:
                self.logger.log({
                    "dagger/iteration": it,
                    "dagger/beta": beta,
                    "dagger/rollout_success_mixed": rollout_success,
                    "dagger/dataset_episodes": self.dataset.num_episodes(),
                    "dagger/dataset_frames": self.dataset.num_frames(),
                    "train/loss_mean": mean_loss,
                    "train/loss_last_epoch": last_loss,
                    "eval/success_rate": success,
                    "eval/mean_return": eval_metrics["mean_return"],
                    "eval/mean_episode_length":
                        eval_metrics["mean_episode_length"],
                }, step=it)

        # Final
        self.save_checkpoint("last")
        print(f"\nDAgger training complete.")
        print(f"  Best in-training success: {self.best_success_rate:.2%}")
        print(f"  Checkpoints in: {self.checkpoint_dir}")