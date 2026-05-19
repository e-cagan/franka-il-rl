"""
Thin Gymnasium wrapper around gymnasium-robotics' FetchPickAndPlace-v4.

Why this wrapper exists:
- The underlying env returns Dict observations (observation, achieved_goal,
  desired_goal). Most IL/RL algorithms (BC, SAC) work more cleanly with
  flat Box observations.
- We flatten obs to a single 28-D vector: 25-D observation + 3-D desired_goal.
- We pass through everything else (action space, reward, termination,
  truncation, info) unchanged.

HER additions (Week 10):
- info["achieved_goal"] is exposed on every reset/step so the HER buffer
  can use it for hindsight relabeling without re-parsing the flat obs.
- compute_reward() proxies to the underlying env so HER can recompute
  rewards against synthetic goals.
- extract_achieved_goal() extracts the cube position from a flat obs;
  used when pre-filling the buffer from a demo HDF5 (where info dicts
  are not stored).

This wrapper is intentionally minimal. The underlying env already handles
physics, IK, goal sampling, and success detection.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import gymnasium_robotics  # noqa: F401 — needed to register the envs


# In FetchPickAndPlace's underlying 25-D observation, cube xyz (= achieved_goal)
# sits at indices [3:6]. This is part of the gymnasium-robotics public layout
# and stable across v3/v4. Codified here so the slice isn't re-discovered
# at multiple call sites.
_CUBE_POS_SLICE = slice(3, 6)


class FetchPickPlaceWrapper(gym.Env):
    """
    Flat-observation wrapper around FetchPickAndPlace-v4.

    Observation (28-D float32):
        - obs[0:25]:   underlying env's "observation" vector
                       (gripper pose, gripper velocity, object pose,
                        object relative to gripper, gripper finger state,
                        object rotation, object velocities — see env docs)
        - obs[25:28]:  desired_goal (target xyz for the object)

    Action (4-D float32, [-1, 1]):
        - action[0:3]: end-effector delta position (dx, dy, dz)
        - action[3]:   gripper command (-1 = close, +1 = open)

    Info dict (added by this wrapper):
        - info["achieved_goal"]: 3-D float32 array, the achieved_goal
          (object xyz) at the current observation. Used by HER.

    The underlying env runs at 25 Hz with max_episode_steps=50, so an
    episode is 2 seconds of sim time. The default reward is sparse: -1
    every step until the goal is reached, then 0.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 25}

    def __init__(self, reward_type="sparse", render_mode=None,
                 max_episode_steps=50):
        super().__init__()

        # The underlying gymnasium-robotics env does all the work
        self._env = gym.make(
            "FetchPickAndPlace-v4",
            reward_type=reward_type,
            render_mode=render_mode,
            max_episode_steps=max_episode_steps,
        )

        # Action space passes through unchanged (4-D Box, [-1, 1])
        self.action_space = self._env.action_space

        # Observation space is flattened: underlying observation + desired_goal
        underlying_obs_shape = self._env.observation_space["observation"].shape[0]
        goal_shape = self._env.observation_space["desired_goal"].shape[0]
        self._obs_dim = underlying_obs_shape + goal_shape
        self._goal_dim = goal_shape

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )

    @property
    def goal_dim(self):
        """Dimensionality of achieved_goal / desired_goal (3 for Fetch)."""
        return self._goal_dim

    @property
    def goal_slice(self):
        """Slice locating desired_goal within the flat observation."""
        return slice(self._obs_dim - self._goal_dim, self._obs_dim)

    @staticmethod
    def extract_achieved_goal(flat_obs):
        """
        Extract achieved_goal (cube xyz) from a flat observation.

        Used when pre-filling the HER buffer from a demo HDF5 — the demo
        format stores obs/action/reward/next_obs/done but does not retain
        info["achieved_goal"]. Since achieved_goal for Fetch is the cube
        position which sits at obs[3:6] of the underlying observation
        (and the wrapper places that underlying obs at flat_obs[0:25]),
        we can recover it by slicing.

        Supports batched input: flat_obs of shape (..., obs_dim) returns
        an array of shape (..., 3).
        """
        return flat_obs[..., _CUBE_POS_SLICE]

    def _flatten_obs(self, dict_obs):
        """Concatenate observation + desired_goal into a single flat array."""
        flat = np.concatenate([
            dict_obs["observation"],
            dict_obs["desired_goal"],
        ]).astype(np.float32)
        return flat

    def _augment_info(self, dict_obs, info):
        """Attach achieved_goal to info for HER consumption."""
        info["achieved_goal"] = dict_obs["achieved_goal"].astype(np.float32).copy()
        return info

    def reset(self, *, seed=None, options=None):
        dict_obs, info = self._env.reset(seed=seed, options=options)
        info = self._augment_info(dict_obs, info)
        return self._flatten_obs(dict_obs), info

    def step(self, action):
        dict_obs, reward, terminated, truncated, info = self._env.step(action)
        info = self._augment_info(dict_obs, info)
        return self._flatten_obs(dict_obs), reward, terminated, truncated, info

    def compute_reward(self, achieved_goal, desired_goal, info=None):
        """
        Proxy to the underlying env's compute_reward.

        Used by HER to compute rewards against relabeled (synthetic) goals.
        Fetch's compute_reward supports batched (N, 3) inputs as well as
        single (3,) inputs.

        Returns:
            For sparse reward: 0.0 on success (distance <= threshold),
            -1.0 otherwise. For dense reward: negative Euclidean distance.
        """
        if info is None:
            info = {}
        return self._env.unwrapped.compute_reward(achieved_goal, desired_goal, info)

    def render(self):
        return self._env.render()

    def close(self):
        self._env.close()

    # --- Convenience accessors for the scripted expert ---
    # The scripted expert needs to read gripper pos, object pos, goal pos
    # from the underlying dict observation. These helpers expose them
    # without parsing the flat vector manually.

    @property
    def unwrapped_env(self):
        """Access the underlying gymnasium-robotics env if needed."""
        return self._env

    def get_state_dict(self):
        """
        Return the underlying dict observation for the current step.
        Useful for the scripted expert which needs structured state access.

        Must be called *after* a reset() or step(); reads from the most
        recent underlying observation.
        """
        unwrapped = self._env.unwrapped
        return unwrapped._get_obs()