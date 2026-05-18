"""
Diagnose stateless expert by comparing it side-by-side with stateful expert
on the same trajectory. For each step, log:
  - stateful expert's phase + action
  - stateless adapter's derived phase + action
  - key geometric quantities (gripper-object, gripper-z above object, etc.)

Reveals where the two diverge.
"""

import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from envs.fetch_pickplace import FetchPickPlaceWrapper
from experts.fetch_expert import FetchExpert
from experts.fetch_expert_stateless import StatelessFetchExpert


def derive_stateless_phase(gripper_pos, object_pos, goal_pos):
    """Mirror StatelessFetchExpert._derive_action's phase decision, but return phase name."""
    xy_dist = np.linalg.norm(gripper_pos[:2] - object_pos[:2])
    z_above_object = gripper_pos[2] - object_pos[2]
    g_to_o = np.linalg.norm(gripper_pos - object_pos)
    o_to_g = np.linalg.norm(object_pos - goal_pos)

    if o_to_g < 0.05:
        return "HOLD_AT_GOAL"
    if g_to_o < 0.03:
        return "GRASP_TRANSPORT"
    if xy_dist < 0.02 and z_above_object > 0.015:
        return "DESCEND"
    return "APPROACH"


def main():
    env = FetchPickPlaceWrapper(render_mode=None)
    stateful = FetchExpert(env)
    stateless = StatelessFetchExpert(env)

    obs, info = env.reset(seed=1000)
    stateful.reset()

    print(f"{'step':>4} | {'stateful_phase':<12} | {'stateless_phase':<15} | "
          f"{'g_to_o':>7} {'xy_dist':>7} {'z_above':>7} {'o_to_g':>7} | "
          f"{'stateful_act':<32} {'stateless_act':<32}")

    for step in range(50):
        state = env.get_state_dict()
        gpos = state["observation"][0:3]
        opos = state["achieved_goal"]
        ggoal = state["desired_goal"]

        g_to_o = np.linalg.norm(gpos - opos)
        xy = np.linalg.norm(gpos[:2] - opos[:2])
        z_above = gpos[2] - opos[2]
        o_to_g = np.linalg.norm(opos - ggoal)

        # Stateful action (will also advance its internal phase)
        sf_phase_name = stateful.phase.name
        sf_action = stateful.act()

        # Reset env-side: undo the act? No — we need stateless's action at SAME state.
        # But stateful.act() already happened. The key: stateless reads state freshly.
        # We just don't apply stateless's action — we apply stateful's, to follow expert trajectory.
        sl_phase_name = derive_stateless_phase(gpos, opos, ggoal)
        sl_action = stateless.act()

        print(f"{step:>4} | {sf_phase_name:<12} | {sl_phase_name:<15} | "
              f"{g_to_o:>7.3f} {xy:>7.3f} {z_above:>7.3f} {o_to_g:>7.3f} | "
              f"{str(sf_action):<32} {str(sl_action):<32}")

        # Apply stateful's action (we want the expert trajectory)
        obs, reward, term, trunc, info = env.step(sf_action)

        if info.get("is_success", 0.0) > 0.5:
            print(f"  -> success at step {step+1}")
            break
        if term or trunc:
            break

    env.close()


if __name__ == "__main__":
    main()