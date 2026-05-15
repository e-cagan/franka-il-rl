"""
Bypass test: skip the action denormalization pipeline.
Write mink's target_q directly to data.ctrl and step MuJoCo manually.
This isolates whether the IK+physics works at all, separate from action pipeline.
"""

import sys
import os

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
import mujoco
from envs.franka_pickplace import FrankaPickPlaceEnv
from experts.ik import FrankaIK

env = FrankaPickPlaceEnv(max_episode_steps=500)
env.reset(seed=0)
ik = FrankaIK(env.model)
env.render()

# Fixed APPROACH target for this test
obj_pos = env.data.xpos[env._object_body_id].copy()
ee_target = obj_pos + np.array([0, 0, 0.10])
grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])  # gripper down

print(f"Object: {obj_pos}")
print(f"EE target (APPROACH): {ee_target}")
print(f"EE initial: {env.data.xpos[env._ee_body_id]}")
print()

# Manual loop, bypassing env.step()
for step in range(500):
    # Solve IK
    target_q = ik.solve(env.data, ee_target, target_quat=grasp_quat)
    
    # Write DIRECTLY to ctrl, no normalization round-trip
    env.data.ctrl[:7] = target_q
    env.data.ctrl[7] = 255.0  # gripper fully open
    
    # Step physics manually (same loop env.step would do)
    for _ in range(env._physics_steps_per_control):
        mujoco.mj_step(env.model, env.data)
    
    env.render()
    time.sleep(0.02)
    
    if step % 25 == 0:
        ee_pos = env.data.xpos[env._ee_body_id]
        current_q = env.data.qpos[:7]
        gap = target_q - current_q
        ee_err = np.linalg.norm(ee_pos - ee_target)
        
        print(f"Step {step}: EE_err={ee_err:.3f}")
        print(f"  current_q:     {current_q.round(3)}")
        print(f"  target_q:      {target_q.round(3)}")
        print(f"  q_gap:         {gap.round(3)}, |gap|={np.linalg.norm(gap):.3f}")
        print(f"  ctrl[:7]:      {env.data.ctrl[:7].round(3)}")
        print(f"  qfrc_actuator: {env.data.qfrc_actuator[:7].round(2)}")
        print(f"  qfrc_bias:     {env.data.qfrc_bias[:7].round(2)}")
        print(f"  qvel[:7]:      {env.data.qvel[:7].round(3)}")

print(f"\nFinal EE: {env.data.xpos[env._ee_body_id]}")
print(f"Final error: {np.linalg.norm(env.data.xpos[env._ee_body_id] - ee_target):.4f}")

env.close()