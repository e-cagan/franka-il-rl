"""
Record a trained policy rollout to GIF + MP4 for showcasing.

Runs the policy in the Fetch env with rgb_array rendering, captures frames,
and writes a single montage GIF (for GitHub README) plus an MP4 (cleaner
for LinkedIn, which re-compresses GIFs poorly). Several successful episodes
from different initial conditions are stitched back-to-back into one clip,
demonstrating that the policy generalizes rather than succeeding by fluke.

Run on the HOST with the project venv (rendering works there); not in the
headless inference container.

Usage:
    python scripts/record_rollout.py \\
        --checkpoint data/checkpoints/bc/bc_seed42/last.pt \\
        --output-dir assets \\
        --num-candidates 12 --keep 4

Produces assets/rollout_montage.gif and .mp4 (a single clip of the `keep`
best successful episodes, separated by a short gap).
"""

import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch

from envs.fetch_pickplace import FetchPickPlaceWrapper
from networks.mlp import MLPPolicy


def load_policy(checkpoint_path, device="cpu"):
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    obs_dim = cfg.get("obs_dim", 28)
    action_dim = cfg.get("action_dim", 4)
    hidden = tuple(cfg.get("hidden_sizes", [256, 256, 256]))
    policy = MLPPolicy(obs_dim, action_dim, hidden).to(device)
    policy.load_state_dict(ckpt["policy_state_dict"])
    policy.eval()
    return policy


@torch.no_grad()
def act(policy, obs, device):
    x = torch.from_numpy(obs).float().unsqueeze(0).to(device)
    a = policy(x).squeeze(0).cpu().numpy().astype(np.float32)
    return np.clip(a, -1.0, 1.0)


def run_episode(env, policy, device, seed):
    """Run one episode, return (frames, success, return, steps)."""
    obs, info = env.reset(seed=seed)
    frames = [env.render()]
    ep_return = 0.0
    success = 0.0
    steps = 0
    done = False
    while not done:
        action = act(policy, obs, device)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(env.render())
        ep_return += float(reward)
        success = float(info.get("is_success", 0.0) > 0.5)
        steps += 1
        done = terminated or truncated
    return frames, success, ep_return, steps


def write_gif(frames, path, fps):
    import imageio
    # imageio infers GIF from extension; duration is per-frame in seconds
    imageio.mimsave(path, frames, duration=1.0 / fps, loop=0)


def write_mp4(frames, path, fps):
    import imageio
    # Requires imageio-ffmpeg; H.264 yuv420p for broad player compatibility
    imageio.mimsave(path, frames, fps=fps, codec="libx264",
                    quality=8, pixelformat="yuv420p")


def make_gap(frame_shape, n_frames):
    """A short run of black frames to separate episodes in the montage."""
    black = np.zeros(frame_shape, dtype=np.uint8)
    return [black.copy() for _ in range(n_frames)]


def hold_last(frames, n_frames):
    """Repeat the final frame to briefly 'pause' on the success state."""
    return [frames[-1].copy() for _ in range(n_frames)]


def build_montage(episodes, gap_frames, hold_frames):
    """
    Stitch episodes into one frame sequence:
        [ep0] [hold on success] [gap] [ep1] [hold] [gap] ... [epN] [hold]
    """
    montage = []
    frame_shape = episodes[0][2][0].shape
    for idx, (_ret, _seed, frames) in enumerate(episodes):
        montage.extend(frames)
        montage.extend(hold_last(frames, hold_frames))
        if idx < len(episodes) - 1:
            montage.extend(make_gap(frame_shape, gap_frames))
    return montage


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="assets")
    parser.add_argument("--num-candidates", type=int, default=12,
                        help="How many episodes to try")
    parser.add_argument("--keep", type=int, default=4,
                        help="How many successful episodes to stitch into the montage")
    parser.add_argument("--seed-start", type=int, default=20000,
                        help="Base seed (disjoint from train/eval seeds)")
    parser.add_argument("--fps", type=int, default=25,
                        help="Playback fps (env runs at 25 Hz)")
    parser.add_argument("--hold-frames", type=int, default=12,
                        help="Frames to pause on each success state (~0.5s at 25fps)")
    parser.add_argument("--gap-frames", type=int, default=6,
                        help="Black frames between episodes (~0.25s at 25fps)")
    parser.add_argument("--diverse", action="store_true", default=True,
                        help="Prefer episodes spanning a range of returns, "
                             "not just the fastest, to show varied cases")
    parser.add_argument("--no-mp4", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = "cpu"
    policy = load_policy(args.checkpoint, device)

    # rgb_array rendering for off-screen frame capture
    env = FetchPickPlaceWrapper(render_mode="rgb_array", max_episode_steps=50)

    successful = []
    for i in range(args.num_candidates):
        seed = args.seed_start + i
        frames, success, ret, steps = run_episode(env, policy, device, seed)
        tag = "OK " if success else "fail"
        print(f"[{tag}] seed={seed}  return={ret:.1f}  steps={steps}  "
              f"frames={len(frames)}")
        if success:
            successful.append((ret, seed, frames))

    env.close()

    if len(successful) < 2:
        raise SystemExit(
            f"Only {len(successful)} successful episode(s); need >=2 for a "
            "montage. Increase --num-candidates."
        )

    # Select `keep` episodes for the montage.
    if args.diverse:
        # Spread across the return range so the clip shows varied cases
        # (different cube/goal placements) rather than N near-identical runs.
        successful.sort(key=lambda c: c[0])  # ascending return
        n = min(args.keep, len(successful))
        idxs = np.linspace(0, len(successful) - 1, n).round().astype(int)
        chosen = [successful[i] for i in idxs]
    else:
        successful.sort(key=lambda c: c[0], reverse=True)
        chosen = successful[:args.keep]

    print(f"\nStitching {len(chosen)} episodes into a montage "
          f"(seeds: {[c[1] for c in chosen]})")
    montage = build_montage(chosen, args.gap_frames, args.hold_frames)
    print(f"Montage length: {len(montage)} frames "
          f"(~{len(montage) / args.fps:.1f}s at {args.fps} fps)")

    gif_path = os.path.join(args.output_dir, "rollout_montage.gif")
    write_gif(montage, gif_path, args.fps)
    print(f"\nSaved {gif_path}")

    if not args.no_mp4:
        mp4_path = os.path.join(args.output_dir, "rollout_montage.mp4")
        try:
            write_mp4(montage, mp4_path, args.fps)
            print(f"Saved {mp4_path}")
        except Exception as e:
            print(f"(mp4 skipped: {e} — pip install imageio-ffmpeg)")

    print("\nReference it in the README, e.g.:")
    print(f"  ![demo]({args.output_dir}/rollout_montage.gif)")


if __name__ == "__main__":
    main()