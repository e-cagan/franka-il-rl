"""
Scripted expert policy for FetchPickAndPlace.

State-machine-based expert that produces demonstrations by reading the
underlying env's structured state (gripper pos, object pos, goal pos)
and emitting 4-D EE delta actions.

Phases:
    APPROACH:  Move gripper above object (5cm above), gripper open.
    DESCEND:   Move gripper down to object level, gripper open.
    GRASP:     Hold position, close gripper, wait for fingers to settle.
    LIFT:      Lift object slightly, then transport toward goal.
    DONE:      Episode complete (success detected by env).
"""

from enum import Enum
import numpy as np


class Phase(Enum):
    APPROACH = 0
    DESCEND = 1
    GRASP = 2
    LIFT = 3
    DONE = 4


class FetchExpert:
    """
    Scripted pick-and-place expert for FetchPickPlaceWrapper.

    Usage:
        env = FetchPickPlaceWrapper(render_mode="human")
        expert = FetchExpert(env)
        obs, _ = env.reset(seed=0)
        expert.reset()
        for step in range(50):
            action = expert.act()
            obs, reward, terminated, truncated, info = env.step(action)
            if expert.is_done() or terminated or truncated:
                break
    """

    def __init__(self, env):
        self.env = env
        self.phase = Phase.APPROACH
        self._phase_step_count = 0

        # Hyperparameters
        self._approach_height = 0.05      # 5cm above object before descending
        self._grasp_threshold = 0.02      # 2cm: close enough to start grasp
        self._descend_threshold = 0.01    # 1cm: close enough vertically
        self._grasp_hold_steps = 4        # steps to hold gripper closed
        self._lift_height = 0.05          # 5cm lift after grasp
        self._action_scale = 5.0          # multiplier on (target - current) deltas

    def reset(self):
        """Call at the start of each new episode."""
        self.phase = Phase.APPROACH
        self._phase_step_count = 0

    def is_done(self):
        return self.phase == Phase.DONE

    def act(self):
        """Compute next 4-D action based on current state and phase."""
        state = self.env.get_state_dict()
        gripper_pos = state["observation"][0:3]
        object_pos = state["achieved_goal"]
        goal_pos = state["desired_goal"]

        ee_delta, gripper_cmd, advance = self._compute_phase_action(
            gripper_pos, object_pos, goal_pos
        )

        if advance:
            self._transition_to_next_phase()

        action = np.concatenate([
            ee_delta,
            np.array([gripper_cmd], dtype=np.float32),
        ]).astype(np.float32)

        self._phase_step_count += 1
        return np.clip(action, -1.0, 1.0)

    def _compute_phase_action(self, gripper_pos, object_pos, goal_pos):
        """
        Returns (ee_delta_xyz, gripper_cmd, should_advance).

        ee_delta is a 3-D delta scaled to roughly fill [-1, 1] for fast motion.
        gripper_cmd: -1 = close, +1 = open.
        """
        if self.phase == Phase.APPROACH:
            target = object_pos + np.array([0.0, 0.0, self._approach_height])
            delta = (target - gripper_pos) * self._action_scale
            gripper_cmd = 1.0  # open
            xy_aligned = np.linalg.norm(gripper_pos[:2] - object_pos[:2]) < self._grasp_threshold
            z_above = gripper_pos[2] > object_pos[2] + self._approach_height - 0.01
            advance = xy_aligned and z_above
            return delta, gripper_cmd, advance

        elif self.phase == Phase.DESCEND:
            target = object_pos
            delta = (target - gripper_pos) * self._action_scale
            gripper_cmd = 1.0  # still open
            advance = np.linalg.norm(gripper_pos - object_pos) < self._descend_threshold
            return delta, gripper_cmd, advance

        elif self.phase == Phase.GRASP:
            delta = np.zeros(3, dtype=np.float32)
            gripper_cmd = -1.0  # close
            advance = self._phase_step_count >= self._grasp_hold_steps
            return delta, gripper_cmd, advance

        elif self.phase == Phase.LIFT:
            # Transport toward goal; lifting happens naturally since goal_z > object_z
            delta = (goal_pos - gripper_pos) * self._action_scale
            gripper_cmd = -1.0  # keep closed
            # Phase doesn't self-advance; env will terminate on success
            advance = False
            return delta, gripper_cmd, advance

        else:  # DONE
            return np.zeros(3, dtype=np.float32), 1.0, False

    def _transition_to_next_phase(self):
        next_phase = Phase(self.phase.value + 1)
        self.phase = next_phase
        self._phase_step_count = 0