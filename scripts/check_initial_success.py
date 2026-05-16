# scripts/check_initial_success.py
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from envs.fetch_pickplace import FetchPickPlaceWrapper

env = FetchPickPlaceWrapper()
initial_distances = []
initial_successes = 0

for seed in range(1000, 1100):
    obs, info = env.reset(seed=seed)
    state = env.get_state_dict()
    obj_pos = state["achieved_goal"]
    goal_pos = state["desired_goal"]
    dist = np.linalg.norm(obj_pos - goal_pos)
    initial_distances.append(dist)
    
    # Check if env considers this initial state a success
    # Take a no-op action just to query info
    action = np.zeros(4, dtype=np.float32)
    obs, reward, term, trunc, info = env.step(action)
    if info.get("is_success", 0) > 0.5:
        initial_successes += 1

env.close()

print(f"Across 100 seeds:")
print(f"  Initial obj-goal distance: min={min(initial_distances):.3f}, "
      f"max={max(initial_distances):.3f}, mean={np.mean(initial_distances):.3f}")
print(f"  Seeds where success was triggered within 1 step: {initial_successes}")
print(f"  Distances < 0.05 (success threshold): "
      f"{sum(1 for d in initial_distances if d < 0.05)}")