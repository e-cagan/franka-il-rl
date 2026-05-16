"""
Policy evaluation harness.

Runs a given policy on an environment for N episodes and reports
aggregate metrics: success rate, mean/std episode return, mean episode
length. Works with any policy that conforms to the
    policy(obs: np.ndarray) -> np.ndarray
interface.
"""

import numpy as np
from collections import defaultdict


class Evaluator:
    """
    Evaluator for policies on a Gymnasium-compatible environment.

    Usage:
        evaluator = Evaluator(env, num_episodes=20)
        metrics = evaluator.evaluate(policy_fn, seed_start=1000)
        print(metrics)
    """

    def __init__(self, env, num_episodes=20, max_episode_steps=None):
        """
        Args:
            env: a Gymnasium-compatible environment instance.
            num_episodes: how many episodes to run per evaluate() call.
            max_episode_steps: optional override; if None, rely on env's truncation.
        """
        self.env = env
        self.num_episodes = num_episodes
        self.max_episode_steps = max_episode_steps

    def evaluate(self, policy_fn, seed_start=1000, reset_callback=None,
                 verbose=False):
        """
        Run policy_fn for num_episodes and return aggregate metrics.

        Args:
            policy_fn: callable obs -> action.
            seed_start: starting seed for episode randomization. Each
                        episode gets seed_start + i so eval is reproducible.
            reset_callback: optional callable invoked after env.reset(),
                            useful for stateful policies that need to be
                            reset per episode (e.g., scripted experts with
                            internal phase state).
            verbose: print per-episode results.

        Returns:
            dict with keys:
                success_rate, mean_return, std_return,
                mean_episode_length, num_episodes,
                returns (list), lengths (list), successes (list)
        """
        returns = []
        lengths = []
        successes = []

        for ep in range(self.num_episodes):
            obs, info = self.env.reset(seed=seed_start + ep)
            if reset_callback is not None:
                reset_callback()

            ep_return = 0.0
            ep_length = 0
            ep_success = 0.0

            while True:
                action = policy_fn(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_return += float(reward)
                ep_length += 1

                # Track success across the episode: once True, stays True
                if info.get("is_success", 0.0) > 0.5:
                    ep_success = 1.0

                # Step limit override
                if self.max_episode_steps is not None and ep_length >= self.max_episode_steps:
                    break
                if terminated or truncated:
                    break

            returns.append(ep_return)
            lengths.append(ep_length)
            successes.append(ep_success)

            if verbose:
                print(f"  ep {ep}: return={ep_return:.2f}, "
                      f"length={ep_length}, success={ep_success}")

        return {
            "success_rate": float(np.mean(successes)),
            "mean_return": float(np.mean(returns)),
            "std_return": float(np.std(returns)),
            "mean_episode_length": float(np.mean(lengths)),
            "num_episodes": self.num_episodes,
            "returns": returns,
            "lengths": lengths,
            "successes": successes,
        }