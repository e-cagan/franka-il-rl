import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from networks.gaussian_policy import GaussianPolicy


def main():
    policy = GaussianPolicy(obs_dim=28, action_dim=4)
    print(f"Policy: {policy}")
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {n_params:,}")

    # Batch forward
    batch_obs = torch.randn(64, 28)
    mean, log_std = policy(batch_obs)
    print(f"\nBatch forward:")
    print(f"  mean shape:    {tuple(mean.shape)}")
    print(f"  log_std shape: {tuple(log_std.shape)}")
    print(f"  log_std range: [{log_std.min():.3f}, {log_std.max():.3f}]")

    # Sampling
    action, log_prob, mean_action = policy.sample(batch_obs)
    print(f"\nSampling:")
    print(f"  action shape:     {tuple(action.shape)}")
    print(f"  action range:     [{action.min():.3f}, {action.max():.3f}]")
    print(f"  log_prob shape:   {tuple(log_prob.shape)}")
    print(f"  log_prob mean:    {log_prob.mean():.3f}")

    # Gradient check: log_prob should flow gradients back to policy params
    policy.train()
    loss = -log_prob.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in policy.parameters())
    print(f"\nGradient check: {has_grad}")

    # Inference (numpy interface)
    obs_np = np.random.randn(28).astype(np.float32)
    action_stoch = policy.act(obs_np, deterministic=False)
    action_det = policy.act(obs_np, deterministic=True)
    print(f"\nNumpy inference:")
    print(f"  stochastic action: {action_stoch}")
    print(f"  deterministic:     {action_det}")
    print(f"  (should differ in noise but be similar in direction)")

    # GPU check
    if torch.cuda.is_available():
        policy = policy.cuda()
        obs_gpu = torch.randn(64, 28).cuda()
        action, log_prob, _ = policy.sample(obs_gpu)
        print(f"\nGPU forward OK, action device: {action.device}")


if __name__ == "__main__":
    main()