import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import time
import numpy as np
from envs.fetch_pickplace import FetchPickPlaceWrapper
from experts.fetch_expert import FetchExpert, Phase


def run_episode(env, expert, render=False, seed=None, verbose=False):
    obs, info = env.reset(seed=seed)
    expert.reset()
    if render:
        env.render()

    total_reward = 0.0
    last_phase = None
    steps = 0

    for step in range(50):
        action = expert.act()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps = step + 1

        if render:
            env.render()
            time.sleep(0.03)

        if verbose and expert.phase != last_phase:
            print(f"  step {step}: phase → {expert.phase.name}")
            last_phase = expert.phase

        if terminated or truncated:
            break

    success = info.get("is_success", 0.0)
    return success, total_reward, steps


def main():
    env = FetchPickPlaceWrapper(render_mode="human")
    expert = FetchExpert(env)

    # First episode: render + verbose
    print("=== Episode 0 (rendered, verbose) ===")
    success, reward, steps = run_episode(env, expert, render=True, seed=0, verbose=True)
    print(f"  result: success={success}, reward={reward:.2f}, steps={steps}")

    # Next 9 episodes: silent, count success rate
    print("\n=== Episodes 1-9 (silent) ===")
    successes = int(success)
    for ep in range(1, 10):
        success, reward, steps = run_episode(env, expert, render=False, seed=ep)
        successes += int(success)
        print(f"  episode {ep}: success={success}, reward={reward:.2f}, steps={steps}")

    print(f"\nSuccess rate: {successes}/10")
    env.close()


if __name__ == "__main__":
    main()