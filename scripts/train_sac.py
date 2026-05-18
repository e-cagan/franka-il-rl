"""
SAC training script for FetchPickAndPlace.

Usage:
    python scripts/train_sac.py --config configs/sac.yaml
    python scripts/train_sac.py --config configs/sac.yaml --total-env-steps 5000 --no-wandb
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
from utils.evaluator import Evaluator
from utils.wandb_logger import WandBLogger
from algos.sac import SACTrainer


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_sac(config, run_name=None, seed=42, use_wandb=True,
              wandb_group=None, verbose=True):
    config = dict(config)
    if run_name is not None:
        config["run_name"] = run_name
    if wandb_group is not None:
        config["wandb_group"] = wandb_group
    config["seed"] = seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    env = FetchPickPlaceWrapper(render_mode=None, reward_type=config.get("reward_type", "sparse"))
    evaluator = Evaluator(env, num_episodes=config.get("eval_episodes", 20))

    logger = None
    if use_wandb:
        logger = WandBLogger(
            project=config.get("wandb_project", "franka-il-rl"),
            run_name=config.get("run_name", "sac_default"),
            config=config,
            tags=config.get("wandb_tags", ["sac"]),
            group=config.get("wandb_group"),
        )

    base_ckpt = config.get("checkpoint_dir", "data/checkpoints/sac")
    run_ckpt = Path(base_ckpt) / config.get("run_name", "sac_default")

    trainer = SACTrainer(
        env=env,
        evaluator=evaluator,
        config=config,
        logger=logger,
        checkpoint_dir=str(run_ckpt),
    )

    try:
        trainer.fit()
    finally:
        env.close()
        if logger is not None:
            logger.finish()

    return {
        "best_success_rate": trainer.best_success_rate,
        "checkpoint_dir": str(run_ckpt),
        "run_name": config.get("run_name"),
        "seed": seed,
        "global_step": trainer.global_step,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/sac.yaml")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--total-env-steps", type=int, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--eval-every-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.total_env_steps is not None:
        config["total_env_steps"] = args.total_env_steps
    if args.warmup_steps is not None:
        config["warmup_steps"] = args.warmup_steps
    if args.eval_every_steps is not None:
        config["eval_every_steps"] = args.eval_every_steps

    train_sac(
        config=config,
        run_name=args.run_name,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()