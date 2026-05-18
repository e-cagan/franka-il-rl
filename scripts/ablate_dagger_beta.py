"""
DAgger β-schedule ablation.

Tests four β schedules on the low-data regime (100-demo init), where
the linear schedule showed high variance (seed 1: 59%, seed 42: 100%).
Hypothesis: a schedule with more expert influence early (e.g. threshold
with k=3) may stabilize seed 1 outliers.

3 seeds × 4 schedules = 12 runs. Each ~10 min. Total ~2 hours.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
from pathlib import Path

from scripts.train_dagger import train_dagger, load_config


# Schedules to sweep
SCHEDULES = [
    {"name": "linear",      "config": {"beta_schedule": "linear"}},
    {"name": "exponential", "config": {"beta_schedule": "exponential",
                                       "beta_decay": 0.7}},
    {"name": "threshold",   "config": {"beta_schedule": "threshold",
                                       "beta_threshold_k": 3}},
    {"name": "constant",    "config": {"beta_schedule": "constant",
                                       "beta_constant": 0.3}},
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/dagger.yaml")
    parser.add_argument("--initial-dataset", type=str,
                        default="data/demonstrations/demos_train_100.hdf5")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--schedules", type=str, nargs="+",
                        default=None,
                        help="Subset of schedules: linear, exponential, threshold, constant")
    parser.add_argument("--group", type=str, default="dagger_beta_ablation")
    parser.add_argument("--results-path", type=str,
                        default="data/checkpoints/dagger/beta_ablation_results.json")
    args = parser.parse_args()

    base_config = load_config(args.config)

    schedule_filter = args.schedules
    schedules_to_run = (SCHEDULES if schedule_filter is None
                        else [s for s in SCHEDULES if s["name"] in schedule_filter])

    all_results = []
    for schedule_def in schedules_to_run:
        name = schedule_def["name"]
        overrides = schedule_def["config"]

        for seed in args.seeds:
            run_name = f"dagger_beta_{name}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"Run: {run_name}")
            print(f"Schedule: {name}, overrides: {overrides}")
            print(f"{'='*60}")

            run_config = dict(base_config)
            run_config["initial_dataset_path"] = args.initial_dataset
            run_config.update(overrides)

            result = train_dagger(
                config=run_config,
                run_name=run_name,
                seed=seed,
                use_wandb=True,
                wandb_group=args.group,
                verbose=True,
            )
            result["schedule"] = name
            result["schedule_overrides"] = overrides
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Beta schedule ablation ({len(all_results)} runs)")
    print(f"{'='*60}")
    print(f"{'schedule':>12} {'seed':>5} {'success (in-train)':>20}")
    for r in all_results:
        print(f"{r['schedule']:>12} {r['seed']:>5} "
              f"{r['best_success_rate']:>19.2%}")

    Path(args.results_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {args.results_path}")


if __name__ == "__main__":
    main()