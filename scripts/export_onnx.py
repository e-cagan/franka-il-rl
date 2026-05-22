"""
Export a trained BC policy checkpoint to ONNX.

Loads the MLP policy from a checkpoint (BCTrainer format), traces it with
a dummy 28-D observation, writes a .onnx file, and verifies numerical
parity against the original PyTorch forward pass.

Usage:
    python scripts/export_onnx.py --checkpoint data/checkpoints/bc/bc_seed42/last.pt
    # writes data/checkpoints/bc/bc_seed42/last.onnx
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch

from networks.mlp import MLPPolicy


def load_policy(checkpoint_path, device="cpu"):
    """Rebuild the MLP policy from a checkpoint's saved config."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if "config" not in ckpt:
        raise ValueError(
            f"{checkpoint_path} has no 'config' field; cannot rebuild "
            "architecture. Was this saved by BCTrainer?"
        )
    cfg = ckpt["config"]
    obs_dim = cfg.get("obs_dim", 28)
    action_dim = cfg.get("action_dim", 4)
    hidden = tuple(cfg.get("hidden_sizes", [256, 256, 256]))

    policy = MLPPolicy(
        obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=hidden,
    ).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, obs_dim, action_dim


def export(checkpoint_path, output_path, opset=17):
    policy, obs_dim, action_dim = load_policy(checkpoint_path)
    dummy = torch.randn(1, obs_dim, dtype=torch.float32)

    torch.onnx.export(
        policy,
        dummy,
        output_path,
        input_names=["observation"],
        output_names=["action"],
        # Dynamic batch axis: the ROS2 node infers one obs at a time, but a
        # flexible batch dim lets the same model be used for batched eval too.
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    print(f"Exported ONNX -> {output_path} "
          f"(obs_dim={obs_dim}, action_dim={action_dim}, opset={opset})")

    verify_parity(policy, output_path, obs_dim)


def verify_parity(policy, onnx_path, obs_dim, n=200, tol=1e-4):
    """Compare PyTorch vs ONNX Runtime output over random observations."""
    import onnxruntime as ort

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    max_diff = 0.0
    for _ in range(n):
        obs = np.random.randn(1, obs_dim).astype(np.float32)
        with torch.no_grad():
            torch_out = policy(torch.from_numpy(obs)).numpy()
        onnx_out = sess.run([out_name], {in_name: obs})[0]
        max_diff = max(max_diff, float(np.abs(torch_out - onnx_out).max()))

    status = "PASS" if max_diff < tol else "FAIL"
    print(f"Parity ({n} samples): max abs diff = {max_diff:.2e} "
          f"(tol {tol:.0e}) -> {status}")
    if max_diff >= tol:
        raise SystemExit("ONNX parity check failed — investigate before deploying")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True,
                        help="Path to BC .pt checkpoint")
    parser.add_argument("--output", default=None,
                        help="Output .onnx path (default: checkpoint with .onnx)")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()

    output = args.output
    if output is None:
        base, _ = os.path.splitext(args.checkpoint)
        output = base + ".onnx"

    export(args.checkpoint, output, opset=args.opset)


if __name__ == "__main__":
    main()