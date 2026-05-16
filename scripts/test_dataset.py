import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from torch.utils.data import DataLoader
from data_utils.demo_dataset import DemoDataset


def main():
    # Load train split
    train_ds = DemoDataset("data/demonstrations/demos_train.hdf5")
    val_ds = DemoDataset("data/demonstrations/demos_val.hdf5")
    test_ds = DemoDataset("data/demonstrations/demos_test.hdf5")

    print(f"Train: {len(train_ds)} frames, {train_ds.num_episodes} episodes")
    print(f"Val:   {len(val_ds)} frames, {val_ds.num_episodes} episodes")
    print(f"Test:  {len(test_ds)} frames, {test_ds.num_episodes} episodes")
    print(f"obs_dim={train_ds.obs_dim}, action_dim={train_ds.action_dim}")

    # Single-item access
    sample = train_ds[0]
    print(f"\nSample 0:")
    for k, v in sample.items():
        if hasattr(v, "shape"):
            print(f"  {k}: shape={tuple(v.shape)}, dtype={v.dtype}")
        else:
            print(f"  {k}: {v}")

    # DataLoader test with batching
    loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    print(f"\nDataLoader: {len(loader)} batches of size 256")

    batch = next(iter(loader))
    print(f"Batch shapes:")
    for k, v in batch.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {tuple(v.shape)}")

    # Episode-level access
    ep = train_ds.get_episode(0)
    print(f"\nEpisode 0:")
    print(f"  length: {len(ep['obs'])}")
    print(f"  obs[0][:5]: {ep['obs'][0][:5]}")
    print(f"  action[0]:  {ep['action'][0]}")

    # Iterate through all batches once (sanity: no crash)
    total_seen = 0
    for batch in loader:
        total_seen += batch["obs"].shape[0]
    print(f"\nFull iteration: {total_seen} frames seen ({total_seen == len(train_ds)})")


if __name__ == "__main__":
    main()