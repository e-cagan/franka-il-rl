# scripts/inspect_random_success.py (geçici)
import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
from envs.fetch_pickplace import FetchPickPlaceWrapper

env = FetchPickPlaceWrapper(render_mode="human")

# Try seeds until we find a random success
for seed in range(1000, 1100):
    obs, info = env.reset(seed=seed)
    env.render()
    ep_success = False
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.02)
        if info.get("is_success", 0) > 0.5:
            ep_success = True
    if ep_success:
        print(f"Seed {seed}: SUCCESS — note what happened")
        time.sleep(3.0)
        break
    else:
        print(f"Seed {seed}: no success")

env.close()