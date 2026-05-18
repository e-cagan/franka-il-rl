import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from networks.q_network import QNetwork
from data_utils.replay_buffer import ReplayBuffer


def main():
    # --- Q-network ---
    print("=== Q-Network ===")
    q = QNetwork(obs_dim=28, action_dim=4)
    n_params = sum(p.numel() for p in q.parameters() if p.requires_grad)
    print(f"Parameters: {n_params:,}")

    batch_obs = torch.randn(64, 28)
    batch_action = torch.randn(64, 4)
    q_vals = q(batch_obs, batch_action)
    print(f"Output shape: {tuple(q_vals.shape)}")
    print(f"Output range (random init): [{q_vals.min():.3f}, {q_vals.max():.3f}]")

    # Gradient check
    loss = q_vals.mean()
    loss.backward()
    has_grad = all(p.grad is not None for p in q.parameters())
    print(f"Gradient flow: {has_grad}")

    # GPU
    if torch.cuda.is_available():
        q.cuda()
        q_gpu = q(batch_obs.cuda(), batch_action.cuda())
        print(f"GPU forward OK, device: {q_gpu.device}")

    # --- Replay buffer ---
    print("\n=== Replay Buffer ===")
    buffer = ReplayBuffer(capacity=1000, obs_dim=28, action_dim=4,
                          device="cuda" if torch.cuda.is_available() else "cpu")
    print(f"Buffer capacity: {buffer.capacity}")
    print(f"Initial size: {len(buffer)}")

    # Add some transitions
    for i in range(500):
        buffer.add(
            obs=np.random.randn(28).astype(np.float32),
            action=np.random.randn(4).astype(np.float32),
            reward=float(np.random.randn()),
            next_obs=np.random.randn(28).astype(np.float32),
            done=bool(np.random.rand() < 0.02),
        )
    print(f"After 500 adds: size={len(buffer)}")

    # Sample
    batch = buffer.sample(256)
    print(f"Batch keys: {list(batch.keys())}")
    for k, v in batch.items():
        print(f"  {k}: shape={tuple(v.shape)}, device={v.device}, dtype={v.dtype}")

    # FIFO wraparound test
    for i in range(700):  # total 500+700 = 1200, capacity 1000 → wrap
        buffer.add(
            obs=np.random.randn(28).astype(np.float32),
            action=np.random.randn(4).astype(np.float32),
            reward=0.0,
            next_obs=np.random.randn(28).astype(np.float32),
            done=False,
        )
    print(f"After 1200 adds (capacity 1000): size={len(buffer)}, "
          f"pos={buffer.pos}")
    print("Wraparound OK" if len(buffer) == 1000 else "WRAPAROUND BUG")

    # End-to-end: sample, run Q-net forward, compute MSE loss, backward
    print("\n=== End-to-end smoke ===")
    q_cpu = QNetwork(obs_dim=28, action_dim=4)
    buffer_cpu = ReplayBuffer(capacity=1000, obs_dim=28, action_dim=4, device="cpu")
    for _ in range(300):
        buffer_cpu.add(
            np.random.randn(28).astype(np.float32),
            np.random.randn(4).astype(np.float32),
            0.0,
            np.random.randn(28).astype(np.float32),
            False,
        )
    batch = buffer_cpu.sample(64)
    pred_q = q_cpu(batch["obs"], batch["action"])
    target = torch.randn_like(pred_q)
    loss = torch.nn.functional.mse_loss(pred_q, target)
    loss.backward()
    print(f"E2E loss: {loss.item():.4f}, gradient flow OK")


if __name__ == "__main__":
    main()