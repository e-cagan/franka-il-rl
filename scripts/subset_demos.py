"""
Create smaller training subsets from the full demonstrations HDF5 for
sample efficiency ablation. Subsets are taken at the episode level
(first N episodes), so trajectory boundaries are preserved.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import argparse
from pathlib import Path
import numpy as np
import h5py


def subset_episodes(src_path, dst_path, num_episodes, seed=42):
    """
    Take a random subset of episodes from src_path, write to dst_path.

    Args:
        src_path: source HDF5 path.
        dst_path: destination HDF5 path.
        num_episodes: how many episodes to include.
        seed: RNG seed for episode selection (reproducible).
    """
    with h5py.File(src_path, "r") as src:
        total_episodes = int(src["metadata"].attrs["num_episodes"])
        if num_episodes > total_episodes:
            raise ValueError(
                f"Requested {num_episodes} episodes but source only has {total_episodes}"
            )

        # Reproducible random selection (not just first N — avoid order bias)
        rng = np.random.default_rng(seed)
        selected = sorted(rng.choice(total_episodes, size=num_episodes, replace=False))

        starts = src["episode_starts"][:]
        lengths = src["episode_lengths"][:]

        # Gather frames from selected episodes
        obs_chunks = []
        act_chunks = []
        rew_chunks = []
        next_obs_chunks = []
        done_chunks = []
        new_starts = []
        new_lengths = []
        cursor = 0

        for ep_idx in selected:
            s = int(starts[ep_idx])
            L = int(lengths[ep_idx])
            obs_chunks.append(src["obs"][s:s + L])
            act_chunks.append(src["action"][s:s + L])
            rew_chunks.append(src["reward"][s:s + L])
            next_obs_chunks.append(src["next_obs"][s:s + L])
            done_chunks.append(src["done"][s:s + L])
            new_starts.append(cursor)
            new_lengths.append(L)
            cursor += L

        obs_arr = np.concatenate(obs_chunks)
        act_arr = np.concatenate(act_chunks)
        rew_arr = np.concatenate(rew_chunks)
        next_obs_arr = np.concatenate(next_obs_chunks)
        done_arr = np.concatenate(done_chunks)

        # Source metadata to copy
        src_meta = dict(src["metadata"].attrs)

    with h5py.File(dst_path, "w") as dst:
        dst.create_dataset("obs", data=obs_arr, compression="gzip")
        dst.create_dataset("action", data=act_arr, compression="gzip")
        dst.create_dataset("reward", data=rew_arr, compression="gzip")
        dst.create_dataset("next_obs", data=next_obs_arr, compression="gzip")
        dst.create_dataset("done", data=done_arr, compression="gzip")
        dst.create_dataset("episode_starts",
                           data=np.array(new_starts, dtype=np.int64))
        dst.create_dataset("episode_lengths",
                           data=np.array(new_lengths, dtype=np.int64))
        dst.create_dataset("success",
                           data=np.ones(len(selected), dtype=bool))

        meta = dst.create_group("metadata")
        for k, v in src_meta.items():
            meta.attrs[k] = v
        meta.attrs["num_episodes"] = len(selected)
        meta.attrs["total_steps"] = int(cursor)
        meta.attrs["subset_of"] = str(src_path)
        meta.attrs["subset_seed"] = seed

    print(f"  {dst_path}: {len(selected)} episodes, {cursor} frames")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="data/demonstrations/demos_train.hdf5")
    parser.add_argument("--output-dir", type=str,
                        default="data/demonstrations")
    parser.add_argument("--counts", type=int, nargs="+",
                        default=[100, 250, 500])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Creating subsets from {args.input}:")
    for n in args.counts:
        dst = output_dir / f"demos_train_{n}.hdf5"
        subset_episodes(args.input, str(dst), n, seed=args.seed)


if __name__ == "__main__":
    main()