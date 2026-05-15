"""
Collect successful demonstrations from FetchExpert and save to HDF5.

Usage:
    python scripts/collect_demos.py --num-episodes 1000 --output data/demonstrations/demos.hdf5
"""

import argparse
import sys
import os
from datetime import datetime
import numpy as np
import h5py

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.fetch_pickplace import FetchPickPlaceWrapper
from experts.fetch_expert import FetchExpert


def collect_one_episode(env, expert, seed):
    """
    Run one episode with the expert. Returns dict of arrays if successful,
    None otherwise.
    """
    obs, info = env.reset(seed=seed)
    expert.reset()
    
    obs_list = [obs]
    action_list = []
    reward_list = []
    next_obs_list = []
    done_list = []
    
    for step in range(50):
        action = expert.act()
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        action_list.append(action)
        reward_list.append(reward)
        next_obs_list.append(next_obs)
        done_list.append(terminated or truncated)
        
        if terminated or truncated:
            break
        
        obs_list.append(next_obs)
    
    success = float(info.get("is_success", 0.0))
    if success < 0.5:
        return None  # Don't save failed episodes
    
    return {
        "obs": np.array(obs_list, dtype=np.float32),
        "action": np.array(action_list, dtype=np.float32),
        "reward": np.array(reward_list, dtype=np.float32),
        "next_obs": np.array(next_obs_list, dtype=np.float32),
        "done": np.array(done_list, dtype=bool),
        "success": True,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-episodes", type=int, default=1000,
                        help="Number of successful episodes to collect")
    parser.add_argument("--output", type=str,
                        default="data/demonstrations/demos.hdf5")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed for episode randomization")
    parser.add_argument("--max-attempts", type=int, default=None,
                        help="Cap on episode attempts (default: 2x num-episodes)")
    args = parser.parse_args()
    
    if args.max_attempts is None:
        args.max_attempts = 2 * args.num_episodes
    
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    env = FetchPickPlaceWrapper(render_mode=None)  # headless for speed
    expert = FetchExpert(env)
    
    # Storage
    all_obs = []
    all_action = []
    all_reward = []
    all_next_obs = []
    all_done = []
    episode_starts = []
    episode_lengths = []
    
    successes = 0
    attempts = 0
    cumulative_steps = 0
    
    while successes < args.num_episodes and attempts < args.max_attempts:
        seed = args.seed_start + attempts
        ep = collect_one_episode(env, expert, seed)
        attempts += 1
        
        if ep is None:
            continue
        
        ep_len = len(ep["action"])
        episode_starts.append(cumulative_steps)
        episode_lengths.append(ep_len)
        all_obs.append(ep["obs"])
        all_action.append(ep["action"])
        all_reward.append(ep["reward"])
        all_next_obs.append(ep["next_obs"])
        all_done.append(ep["done"])
        cumulative_steps += ep_len
        successes += 1
        
        if successes % 50 == 0:
            print(f"Collected {successes}/{args.num_episodes} "
                  f"(attempts: {attempts}, total steps: {cumulative_steps})")
    
    env.close()
    
    # Concatenate
    obs_arr = np.concatenate(all_obs, axis=0)
    action_arr = np.concatenate(all_action, axis=0)
    reward_arr = np.concatenate(all_reward, axis=0)
    next_obs_arr = np.concatenate(all_next_obs, axis=0)
    done_arr = np.concatenate(all_done, axis=0)
    
    # Save to HDF5
    with h5py.File(args.output, "w") as f:
        f.create_dataset("obs", data=obs_arr, compression="gzip")
        f.create_dataset("action", data=action_arr, compression="gzip")
        f.create_dataset("reward", data=reward_arr, compression="gzip")
        f.create_dataset("next_obs", data=next_obs_arr, compression="gzip")
        f.create_dataset("done", data=done_arr, compression="gzip")
        f.create_dataset("episode_starts", data=np.array(episode_starts, dtype=np.int64))
        f.create_dataset("episode_lengths", data=np.array(episode_lengths, dtype=np.int64))
        f.create_dataset("success", data=np.ones(successes, dtype=bool))
        
        meta = f.create_group("metadata")
        meta.attrs["num_episodes"] = successes
        meta.attrs["total_steps"] = cumulative_steps
        meta.attrs["env_id"] = "FetchPickAndPlace-v4"
        meta.attrs["obs_dim"] = obs_arr.shape[1]
        meta.attrs["action_dim"] = action_arr.shape[1]
        meta.attrs["created_at"] = datetime.now().isoformat()
        meta.attrs["expert_class"] = "FetchExpert"
        meta.attrs["success_rate"] = successes / attempts
    
    print(f"\nSaved {successes} episodes ({cumulative_steps} steps) to {args.output}")
    print(f"Success rate: {successes}/{attempts} = {100*successes/attempts:.1f}%")
    print(f"Average episode length: {cumulative_steps/successes:.1f} steps")


if __name__ == "__main__":
    main()