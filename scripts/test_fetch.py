"""
Quick sanity check: load FetchPickAndPlace, render with random policy.
"""

import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
import gymnasium as gym
import gymnasium_robotics  # registers the envs

env = gym.make("FetchPickAndPlace-v4", render_mode="human")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

obs, info = env.reset(seed=0)
print(f"\nObservation structure:")
if isinstance(obs, dict):
    for k, v in obs.items():
        print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
else:
    print(f"  obs shape: {obs.shape}")

print(f"\nRunning 200 steps with random policy...")
for step in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 50 == 0:
        print(f"Step {step}: reward={reward:.3f}, success={info.get('is_success', '?')}")
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        obs, info = env.reset()

env.close()