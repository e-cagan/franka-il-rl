"""
Scripted expert policy for pick-and-place.
Uses a phase-based state machine with mink IK to produce demonstrations.
"""

from enum import Enum
import numpy as np
from experts.ik import FrankaIK


class Phase(Enum):
    APPROACH = 0    # EE 10cm above object, gripper open
    DESCEND = 1     # EE down to grasp height
    GRASP = 2       # close gripper, wait
    LIFT = 3        # lift object 10cm up
    MOVE = 4        # move to above target
    PLACE = 5       # descend to target height
    RELEASE = 6     # open gripper
    DONE = 7        # episode complete


class ScriptedExpert:
    def __init__(self, env):
        self.env = env
        self.ik = FrankaIK(env.model)
        self.phase = Phase.APPROACH
        self._phase_step_count = 0
        # Hyperparameters for phase transitions and waypoints
        self._approach_height = 0.10        # 10cm above object
        self._grasp_height = 0.025          # cube half-size, fingers around it
        self._lift_height = 0.10            # 10cm above table after grasp
        self._place_height = 0.005          # just above target
        self._position_tolerance = 0.015    # 1.5cm to consider waypoint reached
        self._grasp_hold_steps = 10         # how many steps to hold gripper closing
        self._release_hold_steps = 5
        self._grasp_orientation = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz, gripper down
    
    def reset(self):
        """Call this at the start of each new episode."""
        self.phase = Phase.APPROACH
        self._phase_step_count = 0
        
    def act(self):
        """Compute next 8-D action based on current MuJoCo state and phase."""
        ee_pos = self.env.data.xpos[self.env._ee_body_id].copy()
        obj_pos = self.env.data.xpos[self.env._object_body_id].copy()
        target_pos = self.env._target_pos.copy()
        
        # Phase-specific target EE position + gripper command
        ee_target, gripper_cmd, advance = self._compute_phase_target(
            ee_pos, obj_pos, target_pos
        )
        
        if advance:
            self._transition_to_next_phase()
        
        # IK -> joint targets
        target_q = self.ik.solve(
            self.env.data, ee_target, target_quat=self._grasp_orientation
        )
        
        # Normalize to [-1, 1]
        low = self.env._ctrl_range[:7, 0]
        high = self.env._ctrl_range[:7, 1]
        arm_action = 2.0 * (target_q - low) / (high - low) - 1.0
        
        action = np.concatenate([
            arm_action,
            np.array([gripper_cmd], dtype=np.float32),
        ]).astype(np.float32)
        
        self._phase_step_count += 1
        return np.clip(action, -1.0, 1.0)
    
    def _compute_phase_target(self, ee_pos, obj_pos, target_pos):
        """
        Returns (target_ee_xyz, gripper_cmd, should_advance_phase).
        gripper_cmd: -1 = close, +1 = open
        """
        # State machine for every step
        if self.phase == Phase.APPROACH:
            ee_target = obj_pos + np.array([0, 0, self._approach_height])
            gripper_cmd = 1.0  # open
            advance = np.linalg.norm(ee_pos - ee_target) < self._position_tolerance
            return ee_target, gripper_cmd, advance
        
        elif self.phase == Phase.DESCEND:
            ee_target = obj_pos + np.array([0, 0, self._grasp_height])
            gripper_cmd = 1.0  # open
            advance = np.linalg.norm(ee_pos - ee_target) < self._position_tolerance
            return ee_target, gripper_cmd, advance
        
        elif self.phase == Phase.GRASP:
            # Hold position, close gripper, wait for fingers to settle
            ee_target = ee_pos  # stay where we are
            gripper_cmd = -1.0  # close
            advance = self._phase_step_count >= self._grasp_hold_steps
            return ee_target, gripper_cmd, advance
        
        elif self.phase == Phase.LIFT:
            # Lift straight up from grasp position
            ee_target = np.array([obj_pos[0], obj_pos[1], 0.4 + self._lift_height])
            gripper_cmd = -1.0  # keep closed
            advance = np.linalg.norm(ee_pos - ee_target) < self._position_tolerance
            return ee_target, gripper_cmd, advance
        
        elif self.phase == Phase.MOVE:
            ee_target = target_pos + np.array([0, 0, self._lift_height])
            gripper_cmd = -1.0
            advance = np.linalg.norm(ee_pos[:2] - ee_target[:2]) < self._position_tolerance
            return ee_target, gripper_cmd, advance
        
        elif self.phase == Phase.PLACE:
            ee_target = target_pos + np.array([0, 0, self._place_height])
            gripper_cmd = -1.0
            advance = np.linalg.norm(ee_pos - ee_target) < self._position_tolerance
            return ee_target, gripper_cmd, advance
        
        elif self.phase == Phase.RELEASE:
            ee_target = ee_pos  # stay
            gripper_cmd = 1.0  # open
            advance = self._phase_step_count >= self._release_hold_steps
            return ee_target, gripper_cmd, advance
        
        else:  # DONE
            ee_target = ee_pos
            gripper_cmd = 1.0
            advance = False
            return ee_target, gripper_cmd, advance
    
    def _transition_to_next_phase(self):
        """Move to next phase, reset phase step counter."""
        next_phase = Phase(self.phase.value + 1)
        self.phase = next_phase
        self._phase_step_count = 0
    
    def is_done(self):
        return self.phase == Phase.DONE