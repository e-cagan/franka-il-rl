import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
from envs.fetch_pickplace import FetchPickPlaceWrapper
from experts.fetch_expert import FetchExpert
from utils.evaluator import Evaluator


def main():
    env = FetchPickPlaceWrapper(render_mode=None)  # headless for speed
    evaluator = Evaluator(env, num_episodes=20)

    # 1. Random policy baseline (floor)
    print("=== Random policy ===")
    random_policy = lambda obs: env.action_space.sample()
    metrics = evaluator.evaluate(random_policy, seed_start=1000)
    print(f"  success_rate:        {metrics['success_rate']:.2%}")
    print(f"  mean_return:         {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  mean_episode_length: {metrics['mean_episode_length']:.1f}")

    # 2. Expert policy ceiling (this defines what BC can aspire to)
    print("\n=== Expert policy ===")
    expert = FetchExpert(env)
    def expert_policy(obs):
        return expert.act()
    metrics = evaluator.evaluate(
        expert_policy,
        seed_start=1000,
        reset_callback=expert.reset,
    )
    print(f"  success_rate:        {metrics['success_rate']:.2%}")
    print(f"  mean_return:         {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  mean_episode_length: {metrics['mean_episode_length']:.1f}")

    env.close()
    print("\nReference values established for BC comparison.")


if __name__ == "__main__":
    main()