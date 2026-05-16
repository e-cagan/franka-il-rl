"""
Thin wrapper around wandb for consistent experiment logging across
BC, DAgger, and SAC training scripts.

Design goals:
- Single source of truth for project name, entity, default config.
- Optional: can be disabled (mode="disabled") for quick local runs
  without spamming the wandb cloud.
- Consistent metric naming conventions across algorithms.
"""

import os
import wandb


class WandBLogger:
    """
    Wrapper around wandb.init() + wandb.log() with sensible defaults.

    Usage:
        logger = WandBLogger(
            project="franka-il-rl",
            run_name="bc_baseline_seed42",
            config={"lr": 3e-4, "batch_size": 256, ...},
            tags=["bc", "fetch"],
        )
        logger.log({"train/loss": 0.123, "train/epoch": 0}, step=0)
        ...
        logger.finish()
    """

    def __init__(self, project, run_name, config=None, tags=None,
                 entity=None, mode="online", group=None):
        """
        Args:
            project: wandb project name (e.g. "franka-il-rl").
            run_name: human-readable run identifier.
            config: dict of hyperparameters to log.
            tags: list of string tags (e.g. ["bc", "ablation"]).
            entity: wandb username/team. None uses default from `wandb login`.
            mode: "online", "offline", or "disabled".
            group: optional group name (useful for grouping seeds of
                   the same experiment).
        """
        self.run = wandb.init(
            project=project,
            name=run_name,
            config=config or {},
            tags=tags or [],
            entity=entity,
            mode=mode,
            group=group,
            reinit=True,  # allow multiple wandb.init() in same process
        )
        self._step = 0

    def log(self, metrics, step=None):
        """
        Log a dict of metrics. If step is None, uses an internal counter.

        Convention: prefix metric names with the phase:
            - train/loss, train/mse, train/lr
            - val/loss, val/success_rate
            - eval/success_rate, eval/mean_return
        """
        if step is not None:
            self._step = step
        wandb.log(metrics, step=self._step)
        self._step += 1

    def log_config_update(self, updates):
        """Add or update hyperparameters mid-run."""
        wandb.config.update(updates, allow_val_change=True)

    def finish(self):
        """Close the run cleanly. Always call this at script end."""
        wandb.finish()