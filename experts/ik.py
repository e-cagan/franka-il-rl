"""
Inverse kinematics solver for Franka Panda using mink.
Thin wrapper around mink's QP-based differential IK.
"""

import numpy as np
import mink


class FrankaIK:
    """
    Differential IK solver for the Franka arm.
    
    Usage:
        ik = FrankaIK(model)
        target_q = ik.solve(data, target_pos, target_quat)
        # target_q is 7-D, ready to be used as control setpoint
    """
    
    def __init__(self, model, ee_frame_name="hand"):
        self.model = model
        self.configuration = mink.Configuration(model)
        
        self.ee_task = mink.FrameTask(
            frame_name=ee_frame_name,
            frame_type="body",
            position_cost=5.0,
            orientation_cost=0.5,
        )
        self.posture_task = mink.PostureTask(model=model, cost=1e-3)
        self.tasks = [self.ee_task, self.posture_task]

        initial_config = mink.Configuration(model)
        self.posture_task.set_target_from_configuration(initial_config)
        
        self._synced = False
    
    def sync(self, data):
        """Sync internal config with MuJoCo state. Call at episode start."""
        self.configuration.update(data.qpos)
        self.posture_task.set_target_from_configuration(self.configuration)
        self._synced = True
    
    def solve(self, data, target_pos, target_quat=None, dt=0.02, num_iterations=5):
        # Always sync with current MuJoCo state before solving
        self.configuration.update(data.qpos)
        
        if target_quat is None:
            target_quat = self._current_ee_quat(data)
        
        target_pose = mink.SE3.from_rotation_and_translation(
            rotation=mink.SO3(np.asarray(target_quat)),
            translation=np.asarray(target_pos),
        )
        self.ee_task.set_target(target_pose)
        
        for _ in range(num_iterations):
            vel = mink.solve_ik(
                self.configuration, self.tasks, dt, solver="quadprog"
            )
            self.configuration.integrate_inplace(vel, dt)
        
        return self.configuration.q[:7].copy()
    
    def _current_ee_quat(self, data):
        import mujoco
        ee_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        return data.xquat[ee_id].copy()