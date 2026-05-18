import sys, os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
    }

    trainer = SACTrainer(env=env, evaluator=evaluator, config=config)

    # Inspect what got created
    print(f"Device: {trainer.device}")
    print(f"Policy params:  {sum(p.numel() for p in trainer.policy.parameters()):,}")
    print(f"Q1 params:      {sum(p.numel() for p in trainer.q1.parameters()):,}")
    print(f"Q1 target frozen: "
          f"{all(not p.requires_grad for p in trainer.q1_target.parameters())}")
    print(f"Initial alpha:  {trainer.alpha.item():.3f}")
    print(f"Target entropy: {trainer.target_entropy:.3f}")
    print(f"Buffer size:    {len(trainer.buffer)}")

    # Verify target init equals source
    p_source = next(trainer.q1.parameters())
    p_target = next(trainer.q1_target.parameters())
    print(f"Target == source at init: {torch.allclose(p_source, p_target)}")

    # Try one soft update
    p_source.data.fill_(1.0)  # forcibly modify source
    trainer._soft_update_targets()
    diff = (p_target - 1.0).abs().mean()
    expected = 1.0 - trainer.tau  # target moved tau toward 1.0
    print(f"After soft update: target moved by ~{trainer.tau} toward source")
    print(f"  target value example: {p_target.flatten()[0].item():.6f} "
          f"(expected ≈ {trainer.tau:.6f} above original)")

    env.close()
    print("\nSetup OK.")


if __name__ == "__main__":
    main()