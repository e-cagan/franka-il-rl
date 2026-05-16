"""
Behavioral Cloning training script.

Loads expert demonstrations, instantiates MLPPolicy + BCTrainer,
trains for N epochs with periodic env evaluation, logs to W&B.

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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bc.yaml")
    parser.add_argument("--run-name", type=str, default=None,
                        help="Override run name from config")
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override epochs from config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable W&B logging (local debug)")
    args = parser.parse_args()

    # --- Config ---
    config = load_config(args.config)
    if args.epochs is not None:
        config["epochs"] = args.epochs
    if args.run_name is not None:
        config["run_name"] = args.run_name
    config["seed"] = args.seed

    # --- Seed ---
    torch.manual_seed(args.seed)
    import numpy as np
    np.random.seed(args.seed)

    # --- Data ---
    train_ds = DemoDataset(config["train_path"])
    val_ds = DemoDataset(config["val_path"])
    print(f"Train: {len(train_ds)} frames ({train_ds.num_episodes} episodes)")
    print(f"Val:   {len(val_ds)} frames ({val_ds.num_episodes} episodes)")

    # --- Env + Evaluator ---
    env = FetchPickPlaceWrapper(render_mode=None)  # headless
    evaluator = Evaluator(env, num_episodes=config.get("eval_episodes", 20))

    # --- Network ---
    policy = MLPPolicy(
        obs_dim=train_ds.obs_dim,
        action_dim=train_ds.action_dim,
        hidden_sizes=tuple(config.get("hidden_sizes", [256, 256, 256])),
    )
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Policy: {n_params:,} trainable parameters")

    # --- Logger ---
    logger = None
    if not args.no_wandb:
        logger = WandBLogger(
            project=config.get("wandb_project", "franka-il-rl"),
            run_name=config.get("run_name", "bc_default"),
            config=config,
            tags=config.get("wandb_tags", ["bc"]),
            group=config.get("wandb_group"),
        )

    # --- Trainer ---
    trainer = BCTrainer(
        policy=policy,
        train_ds=train_ds,
        val_ds=val_ds,
        evaluator=evaluator,
        config=config,
        logger=logger,
        checkpoint_dir=config.get("checkpoint_dir", "data/checkpoints/bc"),
    )

    try:
        trainer.fit()
    finally:
        env.close()
        if logger is not None:
            logger.finish()


if __name__ == "__main__":
    main()