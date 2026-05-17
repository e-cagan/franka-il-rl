"""
Robust evaluation of trained BC checkpoints.

Runs a much larger eval suite (default 100 episodes) on saved checkpoints
to expose any weakness hidden by the small 20-episode in-training eval.
Reports mean success rate, mean return, and confidence interval per checkpoint.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import json
from pathlib import Path
import numpy as np
import torch

from envs.fetch_pickplace import FetchPickPlaceWrapper
from networks.mlp import MLPPolicy
from utils.evaluator import Evaluator


def load_policy(checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    policy = MLPPolicy(
        obs_dim=28,
        action_dim=4,
        hidden_sizes=tuple(config.get("hidden_sizes", [256, 256, 256])),
    )
    policy.load_state_dict(ckpt["policy_state_dict"])
    return policy, ckpt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Paths to checkpoint .pt files")
    parser.add_argument("--num-episodes", type=int, default=100)
    parser.add_argument("--seed-start", type=int, default=10000,
                        help="Different from training & in-training eval seeds")
    parser.add_argument("--output", type=str,
                        default="data/checkpoints/bc/robust_eval.json")
    args = parser.parse_args()

    env = FetchPickPlaceWrapper(render_mode=None)
    evaluator = Evaluator(env, num_episodes=args.num_episodes)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    all_results = []
    for ckpt_path in args.checkpoints:
        print(f"\n=== Evaluating {ckpt_path} ===")
        policy, ckpt = load_policy(ckpt_path)
        policy.to(device)

        def policy_fn(obs_np):
            return policy.act(obs_np)

        metrics = evaluator.evaluate(policy_fn, seed_start=args.seed_start)

        # 95% CI for success rate via binomial proportion (Wilson would be better
        # but normal approximation is good enough for visualization)
        p = metrics["success_rate"]
        n = args.num_episodes
        se = np.sqrt(p * (1 - p) / n) if n > 0 else 0
        ci_low = max(0.0, p - 1.96 * se)
        ci_high = min(1.0, p + 1.96 * se)

        result = {
            "checkpoint": ckpt_path,
            "num_episodes": args.num_episodes,
            "success_rate": metrics["success_rate"],
            "success_ci_95": [ci_low, ci_high],
            "mean_return": metrics["mean_return"],
            "std_return": metrics["std_return"],
            "mean_episode_length": metrics["mean_episode_length"],
            "num_failures": int(round((1 - p) * n)),
        }
        all_results.append(result)

        print(f"  success: {p:.2%} (95% CI: [{ci_low:.2%}, {ci_high:.2%}])")
        print(f"  failures: {result['num_failures']}/{n}")
        print(f"  return:  {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")

    env.close()

    # Summary table
    print(f"\n{'='*70}")
    print(f"{'Checkpoint':<45} {'Success':>10} {'CI95':>15}")
    print(f"{'-'*70}")
    for r in all_results:
        name = Path(r["checkpoint"]).parent.name
        ci_str = f"[{r['success_ci_95'][0]:.2%}, {r['success_ci_95'][1]:.2%}]"
        print(f"{name:<45} {r['success_rate']:>9.2%} {ci_str:>15}")

    # Save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()