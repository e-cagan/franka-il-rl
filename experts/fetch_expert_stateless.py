"""
Stateless adapter for FetchExpert — NEGATIVE RESULT (Week 8).

Goal: provide a stateless oracle for DAgger queries that re-derives the
appropriate action from observation at each call, avoiding the noisy
labels produced by the stateful FetchExpert's internal phase counter.

Outcome: three iterations of phase-derivation logic were attempted:
  v1: Geometric phases only (xy/z thresholds). Result: 34% success.
      Premature transport — gripper closed and moved toward goal before
      fully grasping the cube.
  v2: Added "object lifted" heuristic (object_z > 0.46). Result: 10%.
      Gripper stayed frozen at object during the implicit "grasp hold"
      window because lifted heuristic triggered too late.
  v3: Used gripper finger separation (obs[9], obs[10]) to detect
      closed-hand state directly. Result: 10%.
      Two issues: (a) initial observation has finger_sep ≈ 0 (env reset
      default), causing first-step false "closed" reading; (b) finger
      separation drops to ~0.022 mid-grasp before the cube is securely
      held, causing premature transport.

The underlying difficulty: scripted state-machine experts have implicit
temporal dependencies (e.g. "hold gripper closed for 4 steps to let
physics settle the grasp") that are not recoverable from any single
observation. A pure stateless re-derivation either misses these
windows entirely or triggers them prematurely.

Future work: a "stateful wrapper" that simulates the FetchExpert
forward from env reset to the queried state would be principled but
prohibitively expensive (would require env state reset + replay per
query). For DAgger, this matters in the high-data regime where BC is
already saturated and stateful expert noise dominates; future
algorithm comparisons in this repo accept this limitation and report
both stateful-DAgger and BC results side by side.

This file is preserved as documentation of the negative result.
Do not import it for active use; FetchExpert (stateful) remains the
DAgger oracle.
"""

import numpy as np


class StatelessFetchExpert:
    """
    Stateless pick-and-place expert.

    Phase decision tree (each act(obs) call):
      1. If gripper is closed (finger separation small):
         -> assume grasping or transporting -> move toward goal, keep closed
      2. If gripper is open AND close to object (in xy and z):
         -> close gripper (start grasp)
      3. If gripper is open AND xy-aligned but above:
         -> descend to object
      4. Otherwise:
         -> approach (move above object)

    No internal state is tracked between calls. The .reset() method is a
    no-op, kept for API compatibility with FetchExpert.

    Args:
        env: FetchPickPlaceWrapper (used only for get_state_dict()).
    """

    # Thresholds
    APPROACH_HEIGHT = 0.05       # target z above object during approach
    XY_ALIGN_THRESHOLD = 0.02    # xy distance below this -> "aligned"
    DESCEND_Z_THRESHOLD = 0.015  # gripper this close in z -> ready to grasp
    GRIPPER_CLOSED_THRESHOLD = 0.035  # mean finger displacement < this -> closed
    ACTION_SCALE = 5.0

    def __init__(self, env):
        self.env = env

    def reset(self):
        """No-op. Kept for API compatibility with FetchExpert."""
        pass

    def is_done(self):
        return False

    def act(self, obs=None):
        """
        Compute next action from current env state.

        Reads gripper finger state from env's underlying observation
        (indices 9 and 10 of the 25-D state["observation"]).
        """
        state = self.env.get_state_dict()
        underlying_obs = state["observation"]
        gripper_pos = underlying_obs[0:3]
        object_pos = state["achieved_goal"]
        goal_pos = state["desired_goal"]
        finger_sep = (underlying_obs[9] + underlying_obs[10]) / 2.0

        ee_delta, gripper_cmd = self._derive_action(
            gripper_pos, object_pos, goal_pos, finger_sep
        )
        action = np.concatenate([
            ee_delta,
            np.array([gripper_cmd], dtype=np.float32),
        ]).astype(np.float32)
        return np.clip(action, -1.0, 1.0)

    def _derive_action(self, gripper_pos, object_pos, goal_pos, finger_sep):
        """
        Pure geometric + gripper-state phase derivation, stateless.
        Returns (ee_delta_xyz, gripper_cmd).
        """
        gripper_closed = finger_sep < self.GRIPPER_CLOSED_THRESHOLD

        # Phase: gripper closed -> grasping or transporting. Move toward goal.
        if gripper_closed:
            delta = (goal_pos - gripper_pos) * self.ACTION_SCALE
            return delta, -1.0

        # Below this point: gripper is OPEN.
        xy_dist = np.linalg.norm(gripper_pos[:2] - object_pos[:2])
        z_above_object = gripper_pos[2] - object_pos[2]
        g_to_o = np.linalg.norm(gripper_pos - object_pos)

        # Phase: gripper open and right at the object -> close to grasp
        if g_to_o < self.DESCEND_Z_THRESHOLD:
            delta = np.zeros(3, dtype=np.float32)
            return delta, -1.0  # close gripper, don't move

        # Phase: gripper xy-aligned, above object -> descend (gripper open)
        if xy_dist < self.XY_ALIGN_THRESHOLD and z_above_object > self.DESCEND_Z_THRESHOLD:
            delta = (object_pos - gripper_pos) * self.ACTION_SCALE
            return delta, 1.0  # stay open

        # Phase: approach -- move above object, gripper open
        target = object_pos + np.array([0.0, 0.0, self.APPROACH_HEIGHT])
        delta = (target - gripper_pos) * self.ACTION_SCALE
        return delta, 1.0