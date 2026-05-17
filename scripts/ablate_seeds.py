"""
Seed sensitivity ablation for BC baseline.

Runs the baseline config across multiple seeds to estimate the variance
of the 100% success result obtained in the initial Week 5 run.
All runs share a W&B group so they appear side-by-side in the dashboard.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/bc.yaml")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 7])
    parser.add_argument("--group", type=str, default="bc_seed_ablation")
    parser.add_argument("--results-path", type=str,
                        default="data/checkpoints/bc/seed_ablation_results.json")
    args = parser.parse_args()

    base_config = load_config(args.config)
    all_results = []

    for seed in args.seeds:
        run_name = f"bc_seed{seed}"
        print(f"\n{'='*60}")
        print(f"Starting run: {run_name}")
        print(f"{'='*60}")

        result = train_bc(
            config=base_config,
            run_name=run_name,
            seed=seed,
            use_wandb=True,
            wandb_group=args.group,
            verbose=True,
        )
        all_results.append(result)

    # Summary
    print(f"\n{'='*60}")
    print(f"Seed ablation summary ({len(all_results)} runs)")
    print(f"{'='*60}")
    success_rates = [r["best_success_rate"] for r in all_results]
    val_losses = [r["best_val_loss"] for r in all_results]
    
    for r in all_results:
        print(f"  seed={r['seed']:3d}: success={r['best_success_rate']:.2%}, "
              f"val_loss={r['best_val_loss']:.4f}")

    import numpy as np
    print(f"\n  success_rate: mean={np.mean(success_rates):.2%}, "
          f"std={np.std(success_rates):.2%}, "
          f"min={min(success_rates):.2%}, "
          f"max={max(success_rates):.2%}")
    print(f"  val_loss:     mean={np.mean(val_losses):.4f}, "
          f"std={np.std(val_losses):.4f}")

    # Save results to JSON for later analysis
    results_path = Path(args.results_path)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {results_path}")


if __name__ == "__main__":
    main()