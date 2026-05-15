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
    for ep in range(5):
        obs, _ = env.reset(seed=ep)
        ep_reward = 0
        ep_len = 0
        for i in range(200):
            action = env.action_space.sample()
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            if term or trunc:
                break
        print(f"Episode {ep}: length={ep_len}, total_reward={ep_reward:.2f}, "
            f"success={info.get('is_success', 0)}")
finally:
    env.close()
    time.sleep(0.2)  # GLFW cleanup