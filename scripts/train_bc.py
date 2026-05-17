"""
Behavioral Cloning training script.

Can be invoked from CLI or imported as a function for ablation scripts.

Usage:
    python scripts/train_bc.py --config configs/bc.yaml
    python scripts/train_bc.py --config configs/bc.yaml --run-name bc_debug --epochs 10
"""

import argparse
import sys
import os
from pathlib import Path

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import yaml
import numpy as np
import torch

from envs.fetch_pickplace import FetchPickPlaceWrapper
from data_utils.demo_dataset import DemoDataset
from networks.mlp import MLPPolicy
from algos.bc import BCTrainer
from utils.evaluator import Evaluator
from utils.wandb_logger import WandBLogger


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_bc(config, run_name=None, seed=42, use_wandb=True,
             wandb_group=None, verbose=True):
    """
    Run one BC training session with given config.

    Args:
        config: dict of hyperparameters.
        run_name: override config["run_name"] if not None.
        seed: random seed.
        use_wandb: if False, no W&B logging.
        wandb_group: override W&B group for ablation grouping.
        verbose: print progress.

    Returns:
        dict with final metrics:
            - best_val_loss
            - best_success_rate
            - final_train_loss
            - final_val_loss
            - checkpoint_dir
    """
    # Resolve run-specific overrides
    config = dict(config)  # shallow copy so we don't mutate caller's config
    if run_name is not None:
        config["run_name"] = run_name
    if wandb_group is not None:
        config["wandb_group"] = wandb_group
    config["seed"] = seed

    # Seed
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_ds = DemoDataset(config["train_path"])
    val_ds = DemoDataset(config["val_path"])
    if verbose:
        print(f"Train: {len(train_ds)} frames ({train_ds.num_episodes} episodes)")
        print(f"Val:   {len(val_ds)} frames ({val_ds.num_episodes} episodes)")

    # Env + Evaluator
    env = FetchPickPlaceWrapper(render_mode=None)
    evaluator = Evaluator(env, num_episodes=config.get("eval_episodes", 20))

    # Network
    policy = MLPPolicy(
        obs_dim=train_ds.obs_dim,
        action_dim=train_ds.action_dim,
        hidden_sizes=tuple(config.get("hidden_sizes", [256, 256, 256])),
    )
    if verbose:
        n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        print(f"Policy: {n_params:,} trainable parameters")

    # Logger
    logger = None
    if use_wandb:
        logger = WandBLogger(
            project=config.get("wandb_project", "franka-il-rl"),
            run_name=config.get("run_name", "bc_default"),
            config=config,
            tags=config.get("wandb_tags", ["bc"]),
            group=config.get("wandb_group"),
        )

    # Per-run checkpoint dir (avoid overwriting across ablation runs)
    base_ckpt_dir = config.get("checkpoint_dir", "data/checkpoints/bc")
    run_ckpt_dir = Path(base_ckpt_dir) / config.get("run_name", "bc_default")

    trainer = BCTrainer(
        policy=policy,
        train_ds=train_ds,
        val_ds=val_ds,
        evaluator=evaluator,
        config=config,
        logger=logger,
        checkpoint_dir=str(run_ckpt_dir),
    )

    try:
        trainer.fit()
    finally:
        env.close()
        if logger is not None:
            logger.finish()

    return {
        "best_val_loss": trainer.best_val_loss,
        "best_success_rate": trainer.best_success_rate,
        "checkpoint_dir": str(run_ckpt_dir),
        "run_name": config.get("run_name"),
        "seed": seed,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bc.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs

    train_bc(
        config=config,
        run_name=args.run_name,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()