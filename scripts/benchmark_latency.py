"""
Benchmark inference latency across backends.

Loads the same policy as PyTorch, ONNX Runtime (CPU/CUDA), and ONNX Runtime
with the TensorRT EP (FP16), then measures per-inference latency over many
single-observation calls (batch=1, matching the deployment regime). Reports
mean / median / p99 latency and throughput. Backends that fail to initialize
(e.g. no GPU, missing TRT) are skipped with a note.

Usage:
    python scripts/benchmark_latency.py \\
        --checkpoint data/checkpoints/bc/bc_seed42/last.pt \\
        --onnx data/checkpoints/bc/bc_seed42/last.onnx

If --onnx is omitted it is derived from --checkpoint. Run scripts/export_onnx.py
first to produce the .onnx file.
"""

import argparse
import os
import sys
import time

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch

from networks.mlp import MLPPolicy


def load_torch_policy(checkpoint_path, device):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    obs_dim = cfg.get("obs_dim", 28)
    action_dim = cfg.get("action_dim", 4)
    hidden = tuple(cfg.get("hidden_sizes", [256, 256, 256]))
    policy = MLPPolicy(obs_dim, action_dim, hidden).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy, obs_dim


def time_fn(infer_fn, obs_samples, warmup, iters):
    """Run warmup then timed iterations; return latencies in milliseconds."""
    for i in range(warmup):
        infer_fn(obs_samples[i % len(obs_samples)])
    lat = np.empty(iters, dtype=np.float64)
    for i in range(iters):
        obs = obs_samples[i % len(obs_samples)]
        t0 = time.perf_counter()
        infer_fn(obs)
        lat[i] = (time.perf_counter() - t0) * 1e3  # ms
    return lat


def report(name, lat):
    print(f"{name:<22} "
          f"mean {lat.mean():7.4f} ms  "
          f"median {np.median(lat):7.4f} ms  "
          f"p99 {np.percentile(lat, 99):7.4f} ms  "
          f"throughput {1000.0 / lat.mean():8.0f} Hz")


def bench_torch(checkpoint, device, obs_samples, warmup, iters):
    policy, _ = load_torch_policy(checkpoint, device)

    @torch.no_grad()
    def infer(obs):
        x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
        out = policy(x)
        return out.squeeze(0).cpu().numpy()

    lat = time_fn(infer, obs_samples, warmup, iters)
    report(f"torch-{device}", lat)


def bench_onnx(onnx_path, providers, label, obs_samples, warmup, iters, trt=False):
    import onnxruntime as ort

    if trt:
        cache = "/tmp/trt_cache_bench"
        os.makedirs(cache, exist_ok=True)
        providers = [
            ("TensorrtExecutionProvider", {
                "trt_fp16_enable": True,
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path": cache,
            }),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]

    sess = ort.InferenceSession(onnx_path, providers=providers)
    active = sess.get_providers()[0]
    in_name = sess.get_inputs()[0].name
    out_name = sess.get_outputs()[0].name

    def infer(obs):
        x = obs.astype(np.float32).reshape(1, -1)
        return sess.run([out_name], {in_name: x})[0]

    lat = time_fn(infer, obs_samples, warmup, iters)
    report(f"{label} [{active.replace('ExecutionProvider', '')}]", lat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--onnx", default=None)
    parser.add_argument("--warmup", type=int, default=200)
    parser.add_argument("--iters", type=int, default=5000)
    parser.add_argument("--obs-dim", type=int, default=28)
    args = parser.parse_args()

    onnx_path = args.onnx
    if onnx_path is None:
        base, _ = os.path.splitext(args.checkpoint)
        onnx_path = base + ".onnx"

    rng = np.random.default_rng(0)
    obs_samples = [rng.standard_normal(args.obs_dim).astype(np.float32)
                   for _ in range(256)]

    print(f"Benchmark: warmup={args.warmup}, iters={args.iters}, batch=1\n")

    # torch CPU (always)
    try:
        bench_torch(args.checkpoint, "cpu", obs_samples, args.warmup, args.iters)
    except Exception as e:
        print(f"torch-cpu skipped: {e}")

    # torch CUDA
    if torch.cuda.is_available():
        try:
            bench_torch(args.checkpoint, "cuda", obs_samples, args.warmup, args.iters)
        except Exception as e:
            print(f"torch-cuda skipped: {e}")

    # ONNX CPU
    try:
        bench_onnx(onnx_path, ["CPUExecutionProvider"], "onnx",
                   obs_samples, args.warmup, args.iters)
    except Exception as e:
        print(f"onnx-cpu skipped: {e}")

    # ONNX CUDA
    try:
        bench_onnx(onnx_path, ["CUDAExecutionProvider", "CPUExecutionProvider"],
                   "onnx", obs_samples, args.warmup, args.iters)
    except Exception as e:
        print(f"onnx-cuda skipped: {e}")

    # TensorRT FP16
    try:
        bench_onnx(onnx_path, None, "tensorrt-fp16",
                   obs_samples, args.warmup, args.iters, trt=True)
    except Exception as e:
        print(f"tensorrt skipped: {e}")


if __name__ == "__main__":
    main()