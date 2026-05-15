"""
FrankaPickPlaceEnv: Gymnasium-compatible MuJoCo environment for pick-and-place
with the Franka Emika Panda arm.
"""

import os
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import mujoco


class FrankaPickPlaceEnv(gym.Env):
    """
    Pick-and-place task: grasp a cube on the table and move it to a target position.
    
    Observation (32-D float32):
        - joint positions (7): Franka arm joint angles
        - joint velocities (7): Franka arm joint velocities
        - end-effector pose (7): EE position (3) + quaternion (4)
        - object pose (7): cube position (3) + quaternion (4)
        - target position (3): goal location for the cube
        - gripper state (1): current finger separation
    
    Action (8-D float32, normalized to [-1, 1]):
        - first 7: joint velocity commands (scaled to actuator limits)
        - last 1: gripper command (-1 = close, +1 = open)
    """
    
    metadata = {"render_modes": ["human"], "render_fps": 20}
    
    def __init__(self, scene_path=None, max_episode_steps=200, control_freq=20):
        super().__init__()
        
        # Resolve paths
        if scene_path is None:
            scene_path = os.path.join("envs", "assets", "pickplace_scene.xml")
        
        menagerie = os.environ.get("MUJOCO_MENAGERIE_PATH")
        if menagerie is None:
            # fallback as before
            repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            menagerie = os.path.join(repo_root, "dev", "mujoco_menagerie")
        panda_dir = os.path.join(menagerie, "franka_emika_panda")
        
        # Load model via temp-file trick
        with open(scene_path) as f:
            xml = f.read()
        
        temp_path = os.path.join(panda_dir, "_env_temp.xml")
        try:
            with open(temp_path, "w") as f:
                f.write(xml)
            self.model = mujoco.MjModel.from_xml_path(temp_path)
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
        
        self.data = mujoco.MjData(self.model)
        
        # Frequency setup
        self._physics_steps_per_control = int(round(
            (1.0 / control_freq) / self.model.opt.timestep
        ))
        
        # Spaces
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(8,), dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(32,), dtype=np.float32
        )
        
        # Cache actuator limits
        self._ctrl_range = self.model.actuator_ctrlrange.copy()
        
        # Cache body/joint IDs
        self._object_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "object"
        )
        self._target_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "target"
        )
        self._object_joint_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_JOINT, "object_freejoint"
        )
        self._ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "hand"
        )
        self._object_qpos_addr = self.model.jnt_qposadr[self._object_joint_id]
        self._target_mocap_id = self.model.body_mocapid[self._target_body_id]
        
        # Episode tracking
        self.max_episode_steps = max_episode_steps
        self._step_count = 0
        
        # Viewer placeholder
        self._viewer = None
    
    def reset(self, *, seed=None, options=None):
        # Handle Gymnasium's RNG properly
        super().reset(seed=seed)
        
        # Reset to keyframe/defaults
        mujoco.mj_resetData(self.model, self.data)
        
        # Randomize object position on the table (within work region)
        obj_x = self.np_random.uniform(0.35, 0.55)
        obj_y = self.np_random.uniform(-0.15, 0.15)
        obj_z = 0.43  # a bit above from table's up surface (half size of the cube is 0.025)
        
        # Randomize target position on the table (different from object pos)
        while True:
            tgt_x = self.np_random.uniform(0.35, 0.55)
            tgt_y = self.np_random.uniform(-0.15, 0.15)
            if np.linalg.norm([tgt_x - obj_x, tgt_y - obj_y]) > 0.10:
                break
        tgt_z = 0.42  # target marker height (Close to initial position in XML)
        
        # Write these into self.data.qpos via the appropriate joint addresses
        self.data.qpos[self._object_qpos_addr + 0] = obj_x
        self.data.qpos[self._object_qpos_addr + 1] = obj_y
        self.data.qpos[self._object_qpos_addr + 2] = obj_z
        # Quaternion (identity = no rotation): w=1, x=y=z=0
        self.data.qpos[self._object_qpos_addr + 3] = 1.0
        self.data.qpos[self._object_qpos_addr + 4] = 0.0
        self.data.qpos[self._object_qpos_addr + 5] = 0.0
        self.data.qpos[self._object_qpos_addr + 6] = 0.0

        # Write target into mocap_pos
        target_mocap_id = self.model.body_mocapid[self._target_body_id]
        self.data.mocap_pos[target_mocap_id] = [tgt_x, tgt_y, tgt_z]

        # Store target pos for reward computation
        self._target_pos = np.array([tgt_x, tgt_y, tgt_z], dtype=np.float32)
        
        # Propagate the kinematics
        mujoco.mj_forward(self.model, self.data)
        
        # Counter and (observation, info) return
        self._step_count = 0
        return self._get_obs(), {}
    
    def step(self, action):
        # Clip and denormalize action
        action = np.clip(action, -1.0, 1.0).astype(np.float32)
        low = self._ctrl_range[:, 0]
        high = self._ctrl_range[:, 1]
        ctrl = low + (action + 1.0) * (high - low) / 2.0
        self.data.ctrl[:] = ctrl
        
        # Advance physics
        for _ in range(self._physics_steps_per_control):
            mujoco.mj_step(self.model, self.data)
        
        # Counter
        self._step_count += 1
        
        # Assemble step return
        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, truncated = self._check_termination()
        info = {}
        
        return obs, reward, terminated, truncated, info
    
    def _get_obs(self):
        """
        Assemble 32-D observation from current MuJoCo state.
        """
        arm_qpos = self.data.qpos[:7]
        arm_qvel = self.data.qvel[:7]
        
        ee_pos = self.data.xpos[self._ee_body_id]
        ee_quat = self.data.xquat[self._ee_body_id]
        
        obj_pos = self.data.xpos[self._object_body_id]
        obj_quat = self.data.xquat[self._object_body_id]
        
        gripper_opening = np.array(
            [self.data.qpos[7] + self.data.qpos[8]], dtype=np.float32
        )
        
        obs = np.concatenate([
            arm_qpos,           # 7
            arm_qvel,           # 7
            ee_pos,             # 3
            ee_quat,            # 4
            obj_pos,            # 3
            obj_quat,           # 4
            self._target_pos,   # 3
            gripper_opening,    # 1
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self):
        # For now: return 0.0 (we'll design reward in Day 4)
        # Just placeholder so step() works
        return 0.0
    
    def _check_termination(self):
        truncated = self._step_count >= self.max_episode_steps
        terminated = False  # success/failure logic (False for now)
        return terminated, truncated
    
    def render(self):
        """Lazy-init passive viewer and sync it."""
        if self._viewer is None:
            from mujoco import viewer
            self._viewer = viewer.launch_passive(self.model, self.data)
        self._viewer.sync()

    def close(self):
        """Clean up viewer if open."""
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None
            import time
            time.sleep(0.1)  # let GLFW finalize