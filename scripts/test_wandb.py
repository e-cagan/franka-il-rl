import sys
import os
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import math
import time
from utils.wandb_logger import WandBLogger


def main():
    # Fake training loop: log dummy metrics to test the pipeline
    logger = WandBLogger(
        project="franka-il-rl",
        run_name="wandb_smoke_test",
        config={
            "lr": 3e-4,
            "batch_size": 256,
            "epochs": 50,
            "obs_dim": 28,
            "action_dim": 4,
        },
        tags=["smoke-test"],
    )

    for epoch in range(50):
        # Simulate decreasing train loss
        train_loss = 1.0 * math.exp(-epoch / 10.0) + 0.05
        val_loss = train_loss + 0.02
        # Simulate increasing success rate during periodic eval
        if epoch % 5 == 0:
            success_rate = min(0.85, 0.1 + epoch * 0.02)
            logger.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "eval/success_rate": success_rate,
                "train/epoch": epoch,
            })
        else:
            logger.log({
                "train/loss": train_loss,
                "val/loss": val_loss,
                "train/epoch": epoch,
            })
        time.sleep(0.05)  # simulate compute

    logger.finish()
    print("\nSmoke test complete. Check the run URL printed by wandb above.")


if __name__ == "__main__":
    main()