"""
Network capacity ablation for BC.

Trains BC with smaller hidden_sizes across multiple seeds. The "baseline"
capacity (3x256) results are reused from the seed ablation (A3).
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


# Capacity configurations to ablate (baseline reused from A3)
CAPACITY_CONFIGS = {
    "small": [128, 128],
    # baseline (3x256) results reused from bc_seed42/1/7
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bc.yaml")
    parser.add_argument("--capacities", type=str, nargs="+",
                        default=["small"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--group", type=str, default="bc_capacity_ablation")
    parser.add_argument("--results-path", type=str,
                        default="data/checkpoints/bc/capacity_results.json")
    args = parser.parse_args()

    base_config = load_config(args.config)
    all_results = []

    for cap_name in args.capacities:
        if cap_name not in CAPACITY_CONFIGS:
            raise ValueError(f"Unknown capacity '{cap_name}'. "
                             f"Available: {list(CAPACITY_CONFIGS.keys())}")
        hidden_sizes = CAPACITY_CONFIGS[cap_name]

        for seed in args.seeds:
            run_name = f"bc_cap_{cap_name}_seed{seed}"
            print(f"\n{'='*60}")
            print(f"Run: {run_name} (hidden_sizes={hidden_sizes})")
            print(f"{'='*60}")

            run_config = dict(base_config)
            run_config["hidden_sizes"] = hidden_sizes

            result = train_bc(
                config=run_config,
                run_name=run_name,
                seed=seed,
                use_wandb=True,
                wandb_group=args.group,
                verbose=True,
            )
            result["capacity"] = cap_name
            result["hidden_sizes"] = hidden_sizes
            all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Capacity ablation summary ({len(all_results)} runs)")
    print(f"{'='*60}")
    print(f"{'capacity':>12} {'seed':>5} {'success':>10} {'val_loss':>10}")
    print("-" * 40)
    for r in all_results:
        print(f"{r['capacity']:>12} {r['seed']:>5} "
              f"{r['best_success_rate']:>9.2%} {r['best_val_loss']:>10.4f}")

    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()