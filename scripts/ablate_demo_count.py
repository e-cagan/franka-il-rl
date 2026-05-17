"""
Sample efficiency ablation for BC.

Trains BC on different demonstration set sizes (100, 250, 500, 800 episodes)
across multiple seeds to estimate how success rate scales with data.
All runs share a W&B group for side-by-side comparison.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
from pathlib import Path

from scripts.train_bc import train_bc, load_config


# Map each demo count to its source HDF5
DEMO_PATHS = {
    100:  "data/demonstrations/demos_train_100.hdf5",
    250:  "data/demonstrations/demos_train_250.hdf5",
    500:  "data/demonstrations/demos_train_500.hdf5",
    800:  "data/demonstrations/demos_train.hdf5",   # full train split
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bc.yaml")
    parser.add_argument("--demo-counts", type=int, nargs="+",
                        default=[100, 250, 500, 800])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--group", type=str, default="bc_demo_count_ablation")
    parser.add_argument("--results-path", type=str,
                        default="data/checkpoints/bc/demo_count_results.json")
    args = parser.parse_args()

    base_config = load_config(args.config)
    all_results = []

    for n_demos in args.demo_counts:
        if n_demos not in DEMO_PATHS:
            raise ValueError(f"No demo path mapped for {n_demos} episodes")
        train_path = DEMO_PATHS[n_demos]

        for seed in args.seeds:
            run_name = f"bc_demos{n_demos}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"Run: {run_name} (train_path={train_path})")
            print(f"{'='*60}")

            # Override train_path for this run
            run_config = dict(base_config)
            run_config["train_path"] = train_path

            result = train_bc(
                config=run_config,
                run_name=run_name,
                seed=seed,
                use_wandb=True,
                wandb_group=args.group,
                verbose=True,
            )
            result["n_demos"] = n_demos
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Demo count ablation summary ({len(all_results)} runs)")
    print(f"{'='*60}")
    print(f"{'n_demos':>8} {'seed':>5} {'success':>10} {'val_loss':>10}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['n_demos']:>8} {r['seed']:>5} "
              f"{r['best_success_rate']:>9.2%} {r['best_val_loss']:>10.4f}")

    # Save
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()