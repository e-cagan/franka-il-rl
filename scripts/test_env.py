import sys
import os

# scripts/'in parent'ı = repo root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
from envs.franka_pickplace import FrankaPickPlaceEnv

env = FrankaPickPlaceEnv()
try:
    obs, _ = env.reset(seed=0)
    env.render()
    time.sleep(1.0)

    for i in range(200):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        time.sleep(0.05)
        if terminated or truncated:
            print(f"Episode ended at step {i}")
            break
finally:
    env.close()
    time.sleep(0.2)  # GLFW cleanup