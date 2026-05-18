"""
DAgger training script.

Loads an initial demonstration set, instantiates MLPPolicy + DAggerTrainer,
runs the DAgger loop (mixed rollouts, expert query, dataset aggregation,
inner BC retrain), logs to W&B.

Usage:
    python scripts/train_dagger.py --config configs/dagger.yaml
    python scripts/train_dagger.py --config configs/dagger.yaml \
        --initial-dataset data/demonstrations/demos_train_100.hdf5 \
        --run-name dagger_init100
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
from experts.fetch_expert import FetchExpert
from networks.mlp import MLPPolicy
from algos.dagger import DAggerTrainer
from utils.evaluator import Evaluator
from utils.wandb_logger import WandBLogger


def load_config(path):
    with open(path) as f:
        return yaml.safe_load(f)


def train_dagger(config, run_name=None, seed=42, use_wandb=True,
                 wandb_group=None, verbose=True):
    """
    Run one DAgger training session.

    Returns:
        dict with best_success_rate, checkpoint_dir, run_name, seed.
    """
    config = dict(config)
    if run_name is not None:
        config["run_name"] = run_name
    if wandb_group is not None:
        config["wandb_group"] = wandb_group
    config["seed"] = seed

    torch.manual_seed(seed)
    np.random.seed(seed)

    # Env + expert + evaluator
    env = FetchPickPlaceWrapper(render_mode=None)
    expert = FetchExpert(env)
    evaluator = Evaluator(env, num_episodes=config.get("eval_episodes", 20))

    # Policy
    policy = MLPPolicy(
        obs_dim=28,
        action_dim=4,
        hidden_sizes=tuple(config.get("hidden_sizes", [256, 256, 256])),
    )
    if verbose:
        n_params = sum(p.numel() for p in policy.parameters()
                       if p.requires_grad)
        print(f"Policy: {n_params:,} trainable parameters")

    # Logger
    logger = None
    if use_wandb:
        logger = WandBLogger(
            project=config.get("wandb_project", "franka-il-rl"),
            run_name=config.get("run_name", "dagger_default"),
            config=config,
            tags=config.get("wandb_tags", ["dagger"]),
            group=config.get("wandb_group"),
        )

    # Per-run checkpoint dir
    base_ckpt = config.get("checkpoint_dir", "data/checkpoints/dagger")
    run_ckpt = Path(base_ckpt) / config.get("run_name", "dagger_default")

    trainer = DAggerTrainer(
        policy=policy,
        env=env,
        expert=expert,
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
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dagger.yaml")
    parser.add_argument("--initial-dataset", type=str, default=None,
                        help="Override initial_dataset_path from config")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--num-iterations", type=int, default=None)
    parser.add_argument("--rollouts-per-iter", type=int, default=None)
    parser.add_argument("--inner-epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.initial_dataset is not None:
        config["initial_dataset_path"] = args.initial_dataset
    if args.num_iterations is not None:
        config["num_iterations"] = args.num_iterations
    if args.rollouts_per_iter is not None:
        config["rollouts_per_iter"] = args.rollouts_per_iter
    if args.inner_epochs is not None:
        config["inner_epochs"] = args.inner_epochs

    train_dagger(
        config=config,
        run_name=args.run_name,
        seed=args.seed,
        use_wandb=not args.no_wandb,
    )


if __name__ == "__main__":
    main()