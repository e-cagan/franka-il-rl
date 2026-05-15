import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from envs.fetch_pickplace import FetchPickPlaceWrapper

env = FetchPickPlaceWrapper(render_mode="human")
print(f"Observation space: {env.observation_space}")
print(f"Action space: {env.action_space}")

obs, info = env.reset(seed=0)
print(f"\nFlat obs shape: {obs.shape}, dtype: {obs.dtype}")
print(f"  obs[0:5]:   {obs[0:5].round(3)}")          # grip pos approx
print(f"  obs[25:28]: {obs[25:28].round(3)}")        # desired goal

# Test that get_state_dict works
state = env.get_state_dict()
print(f"\nState dict keys: {list(state.keys())}")
print(f"  observation shape: {state['observation'].shape}")
print(f"  desired_goal: {state['desired_goal'].round(3)}")
print(f"  achieved_goal: {state['achieved_goal'].round(3)}")

# Sanity: run 50 steps with random action, verify no crash
for step in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if step % 10 == 0:
        print(f"Step {step}: reward={reward:.2f}, success={info.get('is_success', 0)}")
    if terminated or truncated:
        print(f"Episode ended at step {step}")
        break

env.close()