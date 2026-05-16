import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from networks.mlp import MLPPolicy


def main():
    # Instantiate with Fetch dims
    policy = MLPPolicy(obs_dim=28, action_dim=4)
    print(f"Policy: {policy}")
    
    # Parameter count
    n_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    print(f"\nTotal trainable parameters: {n_params:,}")
    
    # Forward pass: batch
    batch_obs = torch.randn(256, 28)
    batch_action = policy(batch_obs)
    print(f"\nBatch forward:")
    print(f"  input  shape: {tuple(batch_obs.shape)}")
    print(f"  output shape: {tuple(batch_action.shape)}")
    print(f"  output range: [{batch_action.min():.3f}, {batch_action.max():.3f}]")
    
    # Inference path (numpy)
    obs_np = np.random.randn(28).astype(np.float32)
    action_np = policy.act(obs_np)
    print(f"\nNumpy inference:")
    print(f"  obs shape:    {obs_np.shape}, dtype={obs_np.dtype}")
    print(f"  action shape: {action_np.shape}, dtype={action_np.dtype}")
    print(f"  action:       {action_np}")
    
    # Gradient check: training mode
    policy.train()
    batch_obs = torch.randn(8, 28, requires_grad=False)
    expert_actions = torch.randn(8, 4).clamp(-1, 1)
    pred_actions = policy(batch_obs)
    loss = torch.nn.functional.mse_loss(pred_actions, expert_actions)
    loss.backward()
    
    has_grad = all(p.grad is not None for p in policy.parameters())
    print(f"\nGradient check: all parameters have gradients: {has_grad}")
    print(f"Sample loss: {loss.item():.4f}")
    
    # GPU check (optional, but worth verifying)
    if torch.cuda.is_available():
        policy = policy.cuda()
        batch_obs_gpu = torch.randn(256, 28).cuda()
        batch_action_gpu = policy(batch_obs_gpu)
        print(f"\nGPU forward pass OK: device={batch_action_gpu.device}, "
              f"shape={tuple(batch_action_gpu.shape)}")
    else:
        print(f"\nNo CUDA available, will train on CPU.")


if __name__ == "__main__":
    main()