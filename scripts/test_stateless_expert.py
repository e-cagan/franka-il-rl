import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from envs.fetch_pickplace import FetchPickPlaceWrapper
from experts.fetch_expert_stateless import StatelessFetchExpert
from utils.evaluator import Evaluator


def main():
    env = FetchPickPlaceWrapper(render_mode=None)
    expert = StatelessFetchExpert(env)
    evaluator = Evaluator(env, num_episodes=50)

    def policy_fn(obs):
        return expert.act(obs)

    metrics = evaluator.evaluate(policy_fn, seed_start=1000)
    print(f"StatelessFetchExpert (50 episodes):")
    print(f"  success_rate:        {metrics['success_rate']:.2%}")
    print(f"  mean_return:         {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  mean_episode_length: {metrics['mean_episode_length']:.1f}")

    env.close()


if __name__ == "__main__":
    main()