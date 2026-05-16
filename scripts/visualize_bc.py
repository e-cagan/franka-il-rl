"""
Load a trained BC checkpoint and render rollouts in the environment.
Quick visual sanity check after training.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
import time
import numpy as np
import torch

from envs.fetch_pickplace import FetchPickPlaceWrapper
from networks.mlp import MLPPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str,
                        default="data/checkpoints/bc/best_success.pt")
    parser.add_argument("--num-episodes", type=int, default=5)
    parser.add_argument("--seed-start", type=int, default=2000,
                        help="Use seeds not seen during training/eval")
    args = parser.parse_args()

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    print(f"Loaded checkpoint: {args.checkpoint}")
    print(f"  Best val_loss: {ckpt['best_val_loss']:.4f}")
    print(f"  Best success rate: {ckpt['best_success_rate']:.2%}")

    # Build env + policy
    env = FetchPickPlaceWrapper(render_mode="human")
    policy = MLPPolicy(
        obs_dim=28,
        action_dim=4,
        hidden_sizes=tuple(config.get("hidden_sizes", [256, 256, 256])),
    )
    policy.load_state_dict(ckpt["policy_state_dict"])

    # Move to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    policy.to(device)

    # Roll out
    successes = 0
    for ep in range(args.num_episodes):
        obs, info = env.reset(seed=args.seed_start + ep)
        env.render()
        time.sleep(0.5)  # let viewer settle

        ep_reward = 0.0
        ep_success = False

        for step in range(50):
            action = policy.act(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            env.render()
            time.sleep(0.04)

            if info.get("is_success", 0.0) > 0.5:
                ep_success = True

            if terminated or truncated:
                break

        successes += int(ep_success)
        print(f"Episode {ep}: success={ep_success}, "
              f"reward={ep_reward:.1f}, "
              f"steps={step+1}")
        time.sleep(0.5)

    print(f"\nVisual eval: {successes}/{args.num_episodes} success")
    env.close()


if __name__ == "__main__":
    main()