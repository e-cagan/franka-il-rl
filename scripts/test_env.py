import sys
import os

# scripts/'in parent'ı = repo root
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
import mujoco
from envs.franka_pickplace import FrankaPickPlaceEnv
from experts.ik import FrankaIK

env = FrankaPickPlaceEnv()
env.reset(seed=0)
ik = FrankaIK(env.model)

env.render()

# Bir dizi hedef pose; her birine git, 1 saniye bekle
targets = [
    [0.4, 0.0, 0.55],   # önde, masaüstü
    [0.5, 0.2, 0.55],   # sağa
    [0.5, -0.2, 0.55],  # sola
    [0.5, 0.0, 0.45],   # aşağı, küpe yakın
]

for tgt in targets:
    print(f"Moving to {tgt}")
    for _ in range(100):  # 100 control step ≈ 5 saniye
        target_q = ik.solve(env.data, tgt)
        
        # Convert target_q to [-1, 1] action
        low = env._ctrl_range[:7, 0]
        high = env._ctrl_range[:7, 1]
        arm_action = 2.0 * (target_q - low) / (high - low) - 1.0
        gripper_action = np.array([1.0])  # open
        action = np.concatenate([arm_action, gripper_action]).astype(np.float32)
        
        env.step(action)
        env.render()
        time.sleep(0.02)
    
    time.sleep(0.5)

env.close()