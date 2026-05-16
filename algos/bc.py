"""
Behavioral Cloning trainer.

Trains a deterministic MLPPolicy to mimic expert actions via MSE loss
on offline demonstration data. Periodically evaluates the policy in
the environment to track real success rate (not just train/val loss).
"""

import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader


class BCTrainer:
    """
    Trainer for Behavioral Cloning.

    Args:
        policy:       MLPPolicy instance (already on target device).
        train_ds:     DemoDataset for training.
        val_ds:       DemoDataset for validation (loss only, no env).
        evaluator:    Evaluator instance for periodic env eval.
        config:       dict with hyperparameters:
                      - lr (default 3e-4)
                      - weight_decay (default 1e-5)
                      - batch_size (default 256)
                      - epochs (default 100)
                      - eval_every (default 5, in epochs)
                      - num_workers (default 0)
                      - device (default "cuda" if available else "cpu")
        logger:       optional WandBLogger.
        checkpoint_dir: where to save best/last checkpoints.
    """

    def __init__(self, policy, train_ds, val_ds, evaluator, config,
                 logger=None, checkpoint_dir="data/checkpoints/bc"):
        self.policy = policy
        self.train_ds = train_ds
        self.val_ds = val_ds
        self.evaluator = evaluator
        self.config = config
        self.logger = logger

        self.device = config.get("device",
                                 "cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.policy.parameters(),
            lr=config.get("lr", 3e-4),
            weight_decay=config.get("weight_decay", 1e-5),
        )

        # Data loaders
        self.batch_size = config.get("batch_size", 256)
        num_workers = config.get("num_workers", 0)
        self.train_loader = DataLoader(
            train_ds, batch_size=self.batch_size,
            shuffle=True, num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )
        self.val_loader = DataLoader(
            val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=num_workers,
            pin_memory=(self.device == "cuda"),
        )

        # Bookkeeping
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_loss = float("inf")
        self.best_success_rate = -1.0
        self.global_step = 0

    def train_epoch(self):
        """Run one full pass over the training set."""
        self.policy.train()
        epoch_losses = []

        for batch in self.train_loader:
            obs = batch["obs"].to(self.device, non_blocking=True)
            expert_action = batch["action"].to(self.device, non_blocking=True)

            pred_action = self.policy(obs)
            loss = F.mse_loss(pred_action, expert_action)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            epoch_losses.append(loss.item())
            self.global_step += 1

        return float(np.mean(epoch_losses))

    @torch.no_grad()
    def validate(self):
        """Compute validation loss (no env interaction)."""
        self.policy.eval()
        val_losses = []
        for batch in self.val_loader:
            obs = batch["obs"].to(self.device, non_blocking=True)
            expert_action = batch["action"].to(self.device, non_blocking=True)
            pred = self.policy(obs)
            loss = F.mse_loss(pred, expert_action)
            val_losses.append(loss.item())
        return float(np.mean(val_losses))

    def evaluate_in_env(self):
        """Roll out the current policy in the env, return metrics dict."""
        self.policy.eval()
        # Adapt MLPPolicy.act for the evaluator interface
        def policy_fn(obs_np):
            return self.policy.act(obs_np)
        return self.evaluator.evaluate(policy_fn)

    def save_checkpoint(self, name):
        """Save a checkpoint with current model + optimizer state."""
        path = self.checkpoint_dir / f"{name}.pt"
        torch.save({
            "policy_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "config": self.config,
            "best_val_loss": self.best_val_loss,
            "best_success_rate": self.best_success_rate,
        }, path)
        return path

    def fit(self):
        """Main training loop."""
        epochs = self.config.get("epochs", 100)
        eval_every = self.config.get("eval_every", 5)

        print(f"Training for {epochs} epochs on {self.device}")
        print(f"Train batches/epoch: {len(self.train_loader)}, "
              f"Val batches: {len(self.val_loader)}")

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_loss = self.validate()

            # Always log losses
            metrics = {
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/epoch": epoch,
            }

            # Periodic env eval
            if (epoch + 1) % eval_every == 0 or epoch == epochs - 1:
                eval_metrics = self.evaluate_in_env()
                metrics.update({
                    "eval/success_rate": eval_metrics["success_rate"],
                    "eval/mean_return": eval_metrics["mean_return"],
                    "eval/mean_episode_length": eval_metrics["mean_episode_length"],
                })

                if eval_metrics["success_rate"] > self.best_success_rate:
                    self.best_success_rate = eval_metrics["success_rate"]
                    self.save_checkpoint("best_success")

                print(f"[epoch {epoch+1:3d}] "
                      f"train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}  "
                      f"success={eval_metrics['success_rate']:.2%}  "
                      f"mean_return={eval_metrics['mean_return']:.1f}")
            else:
                print(f"[epoch {epoch+1:3d}] "
                      f"train_loss={train_loss:.4f}  "
                      f"val_loss={val_loss:.4f}")

            # Best val loss tracking
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best_val")

            if self.logger is not None:
                self.logger.log(metrics, step=epoch)

        # Final checkpoint
        self.save_checkpoint("last")
        print(f"\nTraining complete.")
        print(f"  Best val_loss:     {self.best_val_loss:.4f}")
        print(f"  Best success rate: {self.best_success_rate:.2%}")
        print(f"  Checkpoints in:    {self.checkpoint_dir}")