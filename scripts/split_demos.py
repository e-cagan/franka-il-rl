"""
Split a demonstrations HDF5 into train/val/test by episode boundaries.

Episodes (not frames) are split to prevent data leakage: frames from
the same trajectory must all land in the same split.

Usage:
    python scripts/split_demos.py \
        --input data/demonstrations/demos.hdf5 \
        --output-dir data/demonstrations \
        --train-ratio 0.8 --val-ratio 0.1 --test-ratio 0.1 \
        --seed 42
"""

import argparse
import os
import sys
import numpy as np
import h5py

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def split_indices(num_episodes, ratios, seed):
    """Return three lists of episode indices for train/val/test."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_episodes)
    
    train_ratio, val_ratio, test_ratio = ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    n_train = int(num_episodes * train_ratio)
    n_val = int(num_episodes * val_ratio)
    
    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]
    
    return train_idx, val_idx, test_idx


def extract_split(src_file, episode_indices):
    """
    Extract specific episodes from source HDF5 into in-memory arrays.
    Returns dict of arrays matching the original HDF5 schema.
    """
    starts = src_file["episode_starts"][:]
    lengths = src_file["episode_lengths"][:]
    
    obs_chunks = []
    action_chunks = []
    reward_chunks = []
    next_obs_chunks = []
    done_chunks = []
    new_starts = []
    new_lengths = []
    
    cumulative = 0
    for ep_idx in episode_indices:
        start = starts[ep_idx]
        length = lengths[ep_idx]
        
        obs_chunks.append(src_file["obs"][start:start + length])
        action_chunks.append(src_file["action"][start:start + length])
        reward_chunks.append(src_file["reward"][start:start + length])
        next_obs_chunks.append(src_file["next_obs"][start:start + length])
        done_chunks.append(src_file["done"][start:start + length])
        
        new_starts.append(cumulative)
        new_lengths.append(length)
        cumulative += length
    
    return {
        "obs": np.concatenate(obs_chunks),
        "action": np.concatenate(action_chunks),
        "reward": np.concatenate(reward_chunks),
        "next_obs": np.concatenate(next_obs_chunks),
        "done": np.concatenate(done_chunks),
        "episode_starts": np.array(new_starts, dtype=np.int64),
        "episode_lengths": np.array(new_lengths, dtype=np.int64),
        "success": np.ones(len(episode_indices), dtype=bool),
    }


def save_split(data, output_path, source_meta):
    """Save a split dict to HDF5 with metadata."""
    with h5py.File(output_path, "w") as f:
        f.create_dataset("obs", data=data["obs"], compression="gzip")
        f.create_dataset("action", data=data["action"], compression="gzip")
        f.create_dataset("reward", data=data["reward"], compression="gzip")
        f.create_dataset("next_obs", data=data["next_obs"], compression="gzip")
        f.create_dataset("done", data=data["done"], compression="gzip")
        f.create_dataset("episode_starts", data=data["episode_starts"])
        f.create_dataset("episode_lengths", data=data["episode_lengths"])
        f.create_dataset("success", data=data["success"])
        
        meta = f.create_group("metadata")
        for key, val in source_meta.items():
            meta.attrs[key] = val
        meta.attrs["num_episodes"] = len(data["episode_lengths"])
        meta.attrs["total_steps"] = len(data["obs"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    with h5py.File(args.input, "r") as src:
        num_episodes = int(src["metadata"].attrs["num_episodes"])
        source_meta = dict(src["metadata"].attrs)
        
        train_idx, val_idx, test_idx = split_indices(
            num_episodes,
            (args.train_ratio, args.val_ratio, args.test_ratio),
            args.seed,
        )
        
        print(f"Splitting {num_episodes} episodes:")
        print(f"  train: {len(train_idx)} ({len(train_idx)/num_episodes:.1%})")
        print(f"  val:   {len(val_idx)} ({len(val_idx)/num_episodes:.1%})")
        print(f"  test:  {len(test_idx)} ({len(test_idx)/num_episodes:.1%})")
        
        for name, idx in [("train", train_idx), ("val", val_idx), ("test", test_idx)]:
            data = extract_split(src, idx)
            output_path = os.path.join(args.output_dir, f"demos_{name}.hdf5")
            save_split(data, output_path, source_meta)
            print(f"  saved {name}: {output_path} "
                  f"({len(data['obs'])} frames, {os.path.getsize(output_path)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()