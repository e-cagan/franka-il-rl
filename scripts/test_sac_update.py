import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch
from envs.fetch_pickplace import FetchPickPlaceWrapper
from utils.evaluator import Evaluator
from algos.sac import SACTrainer


def main():
    env = FetchPickPlaceWrapper(render_mode=None)
    evaluator = Evaluator(env, num_episodes=5)

    config = {
        "total_env_steps": 100,
        "warmup_steps": 50,
        "buffer_capacity": 1000,
        "batch_size": 64,  # smaller for quick test
    }

    trainer = SACTrainer(env=env, evaluator=evaluator, config=config)

    # Fill buffer with random transitions
    print("Filling buffer with 200 random transitions...")
    for _ in range(200):
        trainer.buffer.add(
            obs=np.random.randn(28).astype(np.float32),
            action=np.random.uniform(-1, 1, 4).astype(np.float32),
            reward=float(np.random.randn() - 0.5),
            next_obs=np.random.randn(28).astype(np.float32),
            done=bool(np.random.rand() < 0.05),
        )
    print(f"Buffer size: {len(trainer.buffer)}")

    # Pre-update snapshot
    p_q1_before = next(trainer.q1.parameters()).clone()
    p_policy_before = next(trainer.policy.parameters()).clone()
    alpha_before = trainer.alpha.item()

    print(f"\nBefore update:")
    print(f"  alpha: {alpha_before:.6f}")

    # Run several update cycles
    print(f"\n=== Running 20 update cycles ===")
    for i in range(20):
        batch = trainer.buffer.sample(trainer.batch_size)
        critic_metrics = trainer.update_critic(batch)
        actor_metrics, log_prob_d = trainer.update_actor(batch)
        alpha_metrics = trainer.update_alpha(log_prob_d)
        trainer._soft_update_targets()

        if i % 5 == 0:
            print(f"  iter {i:3d}: q1_loss={critic_metrics['q1_loss']:.4f}  "
                  f"actor_loss={actor_metrics['actor_loss']:.4f}  "
                  f"alpha={alpha_metrics['alpha']:.4f}  "
                  f"entropy={actor_metrics['entropy_mean']:.3f}")

    # Verify weights changed
    p_q1_after = next(trainer.q1.parameters())
    p_policy_after = next(trainer.policy.parameters())
    alpha_after = trainer.alpha.item()

    print(f"\nAfter 20 updates:")
    print(f"  Q1 changed:     {not torch.allclose(p_q1_before, p_q1_after)}")
    print(f"  Policy changed: {not torch.allclose(p_policy_before, p_policy_after)}")
    print(f"  Alpha:          {alpha_before:.6f} -> {alpha_after:.6f}")

    # Target should move slowly toward Q1
    p_q1_target = next(trainer.q1_target.parameters())
    diff_target_source = (p_q1_after - p_q1_target).abs().mean()
    print(f"  Target Q1 vs Q1 difference (should be small but >0): "
          f"{diff_target_source:.6f}")

    env.close()
    print("\nUpdate sanity OK.")


if __name__ == "__main__":
    main()